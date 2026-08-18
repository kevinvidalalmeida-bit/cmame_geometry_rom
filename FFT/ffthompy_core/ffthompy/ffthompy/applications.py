import numpy as np
import ffthompy.projections as proj
from ffthompy.tensors.objects import SYM21_PAIRS
from ffthompy.materials import Material
from ffthompy.postprocess import postprocess, add_macro2minimizer
from ffthompy.general.solver import linear_solver
from ffthompy.general.solver_pp import CallBack, CallBack_GA
from ffthompy.general.base import Timer
from ffthompy.tensors import Tensor, DFT, Operator
from ffthompy.tensors.fft import cupy_synchronize, get_array_module, to_backend_array, to_host_array
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import sys
import sysconfig
import time


PROFILE_COMPONENT_LABELS = ("FN", "hG", "FiN", "A")
CG_PROFILE_LABELS = (
    "initialization",
    "Afun",
    "dot_pap_alpha",
    "update_xr",
    "dot_rr_beta",
    "update_p",
    "residual_check",
    "callback",
)


# A projection depends on the grid, cell and discretization, but not on the
# material coefficients.  Adaptive fixed-geometry campaigns can therefore
# retain it, including its lazily-created CuPy array, across material solves.
_ELASTICITY_PROJECTION_CACHE = {}


def clear_elasticity_projection_cache():
    """Release session-owned Fourier projection tensors."""
    _ELASTICITY_PROJECTION_CACHE.clear()


def _elasticity_projection_cache_key(
    pb, *, fft_form, real_dtype, projection_storage, projection_backend
):
    return (
        tuple(int(value) for value in pb.solve['N']),
        tuple(float(value) for value in pb.Y),
        str(pb.solve.get('kind')),
        str(fft_form),
        np.dtype(real_dtype).str,
        str(projection_storage),
        str(projection_backend),
    )


def _solver_get(conf, key, default=None):
    if isinstance(conf, dict):
        return conf.get(key, default)
    return getattr(conf, key, default)


def _solver_callback_mode(pb):
    mode = _solver_get(pb.solver, "callback", "none")
    if mode is None:
        return "none"
    if isinstance(mode, bool):
        return "residual" if mode else "none"
    mode = str(mode).strip().lower()
    aliases = {
        "": "none",
        "0": "none",
        "false": "none",
        "off": "none",
        "no": "none",
        "none": "none",
        "disabled": "none",
        "1": "residual",
        "true": "residual",
        "on": "residual",
        "yes": "residual",
        "basic": "residual",
        "res": "residual",
        "residual": "residual",
        "detailed": "detailed",
    }
    if mode not in aliases:
        raise NotImplementedError("The solver callback (%s) is not implemented" % mode)
    return aliases[mode]


def _make_solver_callback(mode, *, Afun, B, EN=None, GN=None, A_Ga=None):
    if mode == "none":
        return None
    if mode == "residual":
        return CallBack(A=Afun, B=B)
    if mode == "detailed":
        return CallBack_GA(A=Afun, B=B, E2N=EN, Aex=A_Ga, GN=GN)
    raise NotImplementedError("The solver callback (%s) is not implemented" % mode)


def _new_component_stats():
    return {label: {"seconds": 0.0, "calls": 0} for label in PROFILE_COMPONENT_LABELS}


class TimedCallable:
    def __init__(self, op, label, stats):
        self.op = op
        self.label = label
        self.stats = stats
        self.name = getattr(op, "name", label)

    def __call__(self, x):
        cupy_synchronize()
        t0 = time.perf_counter()
        out = self.op(x)
        cupy_synchronize(out)
        dt = time.perf_counter() - t0
        self.stats[self.label]["seconds"] += dt
        self.stats[self.label]["calls"] += 1
        return out

    def transpose(self):
        if hasattr(self.op, "transpose"):
            return TimedCallable(self.op.transpose(), self.label, self.stats)
        raise AttributeError(f"'{type(self.op).__name__}' object has no attribute 'transpose'")

    def __getattr__(self, item):
        return getattr(self.op, item)


def _profile_enabled(pb):
    return bool(pb.solve.get("profile_timing", False))


def _stage_tic():
    cupy_synchronize()
    return time.perf_counter()


def _stage_toc(t0):
    cupy_synchronize()
    return time.perf_counter() - t0


def _fast_macro_add_enabled(pb):
    return bool(_solver_get(pb.solver, "fast_macro_add", pb.solve.get("fast_macro_add", False)))


def _check_macro_mean_enabled(pb):
    return bool(_solver_get(pb.solver, "check_macro_mean", pb.solve.get("check_macro_mean", True)))


def _projection_storage(pb):
    return str(pb.solve.get("projection_storage", "full")).strip().lower()


def _projection_backend(pb):
    return str(pb.solve.get("projection_backend", "auto")).strip().lower()


def _pack_tensor_sym21(tensor):
    val = np.stack([tensor.val[ii, jj] for ii, jj in SYM21_PAIRS], axis=0)
    return tensor.copy(
        name=tensor.name + '_sym21',
        val=val,
        order=1,
        multype='sym21',
    )


def _load_batch_size(pb, nloads):
    if _profile_enabled(pb):
        return 1
    size = int(_solver_get(pb.solver, "load_batch_size", pb.solve.get("load_batch_size", 1)))
    return max(1, min(size, int(nloads)))


def _solver_real_dtype(pb):
    dtype = _solver_get(pb.solver, "real_dtype", None)
    if dtype is None:
        dtype = _solver_get(pb.solve, "real_dtype", None)
    if dtype is None:
        dtype = "float32" if _solver_get(pb.solve, "fft_backend", "") == "cupy" else "float64"
    return np.dtype(dtype)


def _instrument_afun_for_profile(Afun, GN, A, stats):
    if len(getattr(GN, "mat_rev", [])) != 1 or len(GN.mat_rev[0]) != 3:
        return Afun

    timed_FN = TimedCallable(GN.mat_rev[0][0], "FN", stats)
    timed_hG = TimedCallable(GN.mat_rev[0][1], "hG", stats)
    timed_FiN = TimedCallable(GN.mat_rev[0][2], "FiN", stats)
    timed_GN = Operator(name=GN.name, mat_rev=[[timed_FN, timed_hG, timed_FiN]])
    timed_A = TimedCallable(A, "A", stats)
    return Operator(name=Afun.name, mat_rev=[[timed_A, timed_GN]])


def _extract_perf_counter_time(info):
    try:
        return float(info["time"][-1][1])
    except Exception:
        return None


def _build_load_timing_payload(iL, load_wall_s, rhs_build_s, info, stats):
    payload = {
        "load_id": int(iL),
        "load_wall_s": float(load_wall_s),
        "rhs_build_s": float(rhs_build_s),
        "linear_solver_perf_counter_s": _extract_perf_counter_time(info),
        "cg_iterations": int(info.get("kit", 0)),
        "final_norm_res": float(info.get("norm_res", np.nan)),
        "final_norm_res_rel": float(info.get("norm_res_rel", np.nan)),
        "rhs_norm": float(info.get("rhs_norm", np.nan)),
        "converged": bool(info.get("converged", False)),
        "cg_profile": info.get("cg_profile", {}),
        "components": {
            label: {
                "seconds": float(stats[label]["seconds"]),
                "calls": int(stats[label]["calls"]),
            }
            for label in PROFILE_COMPONENT_LABELS
        },
    }
    return payload


def _solver_info_payload(info, load_ids):
    norm_per_rhs = info.get("norm_res_per_rhs", None)
    if norm_per_rhs is not None:
        norm_per_rhs = [float(value) for value in np.asarray(norm_per_rhs).ravel()]
    rel_per_rhs = info.get("norm_res_rel_per_rhs", None)
    if rel_per_rhs is not None:
        rel_per_rhs = [float(value) for value in np.asarray(rel_per_rhs).ravel()]
    rhs_norm_per_rhs = info.get("rhs_norm_per_rhs", None)
    if rhs_norm_per_rhs is not None:
        rhs_norm_per_rhs = [float(value) for value in np.asarray(rhs_norm_per_rhs).ravel()]
    converged_per_rhs = info.get("converged_per_rhs", None)
    if converged_per_rhs is not None:
        converged_per_rhs = [bool(value) for value in np.asarray(converged_per_rhs).ravel()]

    payload = {
        "load_ids": [int(value) for value in load_ids],
        "cg_iterations": int(info.get("kit", 0)),
        "final_norm_res": float(info.get("norm_res", np.nan)),
        "final_norm_res_rel": float(info.get("norm_res_rel", np.nan)),
        "rhs_norm": float(info.get("rhs_norm", np.nan)),
        "threshold": float(info.get("threshold", np.nan)),
        "converged": bool(info.get("converged", False)),
        "hit_maxiter": bool(info.get("hit_maxiter", False)),
    }
    if norm_per_rhs is not None:
        payload["final_norm_res_per_rhs"] = norm_per_rhs
    if rel_per_rhs is not None:
        payload["final_norm_res_rel_per_rhs"] = rel_per_rhs
    if rhs_norm_per_rhs is not None:
        payload["rhs_norm_per_rhs"] = rhs_norm_per_rhs
    if converged_per_rhs is not None:
        payload["converged_per_rhs"] = converged_per_rhs
    return payload


def _summarize_solver_infos(load_solver_infos, rtol, atol, maxiter):
    norms = []
    relative_norms = []
    iterations = []
    convergence_flags = []
    load_count = 0
    for item in load_solver_infos:
        load_count += len(item.get("load_ids", []))
        iterations.append(int(item.get("cg_iterations", 0)))
        if "final_norm_res_per_rhs" in item:
            norms.extend(float(value) for value in item["final_norm_res_per_rhs"])
        else:
            norms.append(float(item.get("final_norm_res", np.nan)))
        if "final_norm_res_rel_per_rhs" in item:
            relative_norms.extend(float(value) for value in item["final_norm_res_rel_per_rhs"])
        else:
            relative_norms.append(float(item.get("final_norm_res_rel", np.nan)))
        if "converged_per_rhs" in item:
            convergence_flags.extend(bool(value) for value in item["converged_per_rhs"])
        else:
            convergence_flags.append(bool(item.get("converged", False)))

    finite_norms = [value for value in norms if np.isfinite(value)]
    finite_relative_norms = [value for value in relative_norms if np.isfinite(value)]
    return {
        "batch_count": int(len(load_solver_infos)),
        "load_count": int(load_count),
        "cg_iterations_sum_batches": int(sum(iterations)),
        "cg_iterations_max": int(max(iterations, default=0)),
        "final_norm_res_max": float(max(finite_norms, default=np.nan)),
        "final_norm_res_mean": float(np.mean(finite_norms)) if finite_norms else np.nan,
        "final_norm_res_rel_max": float(max(finite_relative_norms, default=np.nan)),
        "final_norm_res_rel_mean": float(np.mean(finite_relative_norms)) if finite_relative_norms else np.nan,
        "rtol": float(rtol),
        "atol": float(atol),
        "tol": float(rtol),
        "maxiter": int(maxiter),
        "all_converged": bool(convergence_flags and all(convergence_flags)),
        "converged_load_count": int(sum(convergence_flags)),
        "hit_maxiter": bool(any(item.get("hit_maxiter", False) for item in load_solver_infos)),
    }


def _aggregate_load_timings(load_timings, primaldual, parallel, backend, workers, primal_dual_perf_counter_s):
    aggregate = {
        "load_count": int(len(load_timings)),
        "load_wall_s_sum": float(sum(item["load_wall_s"] for item in load_timings)),
        "load_wall_s_mean": float(np.mean([item["load_wall_s"] for item in load_timings])) if load_timings else 0.0,
        "rhs_build_s_sum": float(sum(item["rhs_build_s"] for item in load_timings)),
        "linear_solver_perf_counter_s_sum": float(
            sum((item["linear_solver_perf_counter_s"] or 0.0) for item in load_timings)
        ),
        "cg_iterations_sum": int(sum(item["cg_iterations"] for item in load_timings)),
        "cg_iterations_max": int(max((item["cg_iterations"] for item in load_timings), default=0)),
        "final_norm_res_max": float(max((item["final_norm_res"] for item in load_timings), default=np.nan)),
        "final_norm_res_mean": float(np.mean([item["final_norm_res"] for item in load_timings])) if load_timings else np.nan,
    }

    for label in PROFILE_COMPONENT_LABELS:
        aggregate[f"{label}_s_sum"] = float(sum(item["components"][label]["seconds"] for item in load_timings))
        aggregate[f"{label}_calls_sum"] = int(sum(item["components"][label]["calls"] for item in load_timings))

    for label in CG_PROFILE_LABELS:
        aggregate[f"CG_{label}_s_sum"] = float(
            sum(item.get("cg_profile", {}).get(label, {}).get("seconds", 0.0) for item in load_timings)
        )
        aggregate[f"CG_{label}_calls_sum"] = int(
            sum(item.get("cg_profile", {}).get(label, {}).get("calls", 0) for item in load_timings)
        )

    return {
        "primaldual": primaldual,
        "parallel": bool(parallel),
        "backend": backend,
        "workers": int(workers),
        "primal_dual_perf_counter_s": float(primal_dual_perf_counter_s),
        "loads": load_timings,
        "aggregate": aggregate,
    }


def _print_timing_summary(profile_summary):
    agg = profile_summary["aggregate"]
    print(
        "[PROFILE] {} | loads={} | FN={:.3f}s | hG={:.3f}s | FiN={:.3f}s | A={:.3f}s | "
        "CG_Afun={:.3f}s | CG_reductions={:.3f}s | CG_updates={:.3f}s | lin_solver={:.3f}s".format(
            profile_summary["primaldual"],
            agg["load_count"],
            agg["FN_s_sum"],
            agg["hG_s_sum"],
            agg["FiN_s_sum"],
            agg["A_s_sum"],
            agg["CG_Afun_s_sum"],
            agg["CG_dot_pap_alpha_s_sum"] + agg["CG_dot_rr_beta_s_sum"],
            agg["CG_update_xr_s_sum"] + agg["CG_update_p_s_sum"],
            agg["linear_solver_perf_counter_s_sum"],
        )
    )


def solve_load_scalar(iL, dim, Nbar, Afun, pb, GN, A, add_macro2minimizer, linear_solver, CallBack, CallBack_GA, fft_form='c'):
    real_dtype = _solver_real_dtype(pb)
    E = np.zeros(dim, dtype=real_dtype)
    E[iL] = 1
    print(('macroscopic load E = ' + str(E)))
    EN = Tensor(name='EN', N=Nbar, shape=(dim,), Fourier=False, fft_form=fft_form, dtype=real_dtype)
    EN.set_mean(E)
    x0 = Tensor(name='x0', N=Nbar, shape=(dim,), Fourier=False, fft_form=fft_form, dtype=real_dtype)
    profile_enabled = _profile_enabled(pb)
    timing_stats = _new_component_stats() if profile_enabled else None
    Afun_local = _instrument_afun_for_profile(Afun, GN, A, timing_stats) if profile_enabled else Afun
    rhs_t0 = time.perf_counter()
    B = Afun_local(-EN)
    cupy_synchronize(B)
    rhs_build_s = time.perf_counter() - rhs_t0
    cb = _make_solver_callback(
        _solver_callback_mode(pb),
        Afun=Afun_local,
        B=B,
        EN=EN,
        GN=GN,
        A_Ga=A,
    )
    load_t0 = time.perf_counter()
    X, info = linear_solver(solver=pb.solver['kind'], Afun=Afun_local, B=B, x0=x0, par=pb.solver, callback=cb)
    load_wall_s = time.perf_counter() - load_t0
    result = {'cb': cb, 'info': info}
    if profile_enabled:
        result['timing'] = _build_load_timing_payload(iL, load_wall_s, rhs_build_s, info, timing_stats)
    return iL, add_macro2minimizer(
        X,
        E,
        check_mean=_check_macro_mean_enabled(pb),
        inplace=_fast_macro_add_enabled(pb),
    ), result


def solve_load_elasticity(iL, D, Nbar, Afun, pb, GN, A, add_macro2minimizer, linear_solver, CallBack, CallBack_GA, fft_form='c'):
    real_dtype = _solver_real_dtype(pb)
    E = np.zeros(D, dtype=real_dtype)
    E[iL] = 1
    print(('macroscopic load E = ' + str(E)))
    EN = Tensor(name='EN', N=Nbar, shape=(D,), Fourier=False, fft_form=fft_form, dtype=real_dtype)
    EN.set_mean(E)
    init_fields = pb.solve.get('initial_solution_fields', None)
    if init_fields is not None:
        x0 = EN.zeros_like(name='x0')
        val = np.asarray(init_fields[iL], dtype=real_dtype)
        if hasattr(x0.val, "get") or type(x0.val).__module__.startswith("cupy"):
            import cupy as cp
            x0.val = cp.asarray(val)
        else:
            x0.val = val
    else:
        x0 = EN.zeros_like(name='x0')
    profile_enabled = _profile_enabled(pb)
    timing_stats = _new_component_stats() if profile_enabled else None
    Afun_local = _instrument_afun_for_profile(Afun, GN, A, timing_stats) if profile_enabled else Afun
    rhs_t0 = time.perf_counter()
    B = Afun_local(-EN)
    cupy_synchronize(B)
    rhs_build_s = time.perf_counter() - rhs_t0
    cb = _make_solver_callback(
        _solver_callback_mode(pb),
        Afun=Afun_local,
        B=B,
        EN=EN,
        GN=GN,
        A_Ga=A,
    )
    load_t0 = time.perf_counter()
    X, info = linear_solver(solver=pb.solver['kind'], Afun=Afun_local, B=B, x0=x0, par=pb.solver, callback=cb)
    load_wall_s = time.perf_counter() - load_t0
    result = {'cb': cb, 'info': info}
    if profile_enabled:
        result['timing'] = _build_load_timing_payload(iL, load_wall_s, rhs_build_s, info, timing_stats)
    return iL, add_macro2minimizer(
        X,
        E,
        check_mean=_check_macro_mean_enabled(pb),
        inplace=_fast_macro_add_enabled(pb),
    ), result


def _split_batched_solution(X, load_ids):
    solutions = []
    for local_id, load_id in enumerate(load_ids):
        val = X.val[:, local_id, ...]
        solutions.append(
            X.copy(
                name='load_{0}'.format(int(load_id)),
                val=val,
                order=1,
            )
        )
    return solutions


def solve_load_elasticity_batch(load_ids, D, Nbar, Afun, pb, GN, A, add_macro2minimizer, linear_solver, CallBack, CallBack_GA, fft_form='c'):
    real_dtype = _solver_real_dtype(pb)
    load_ids = [int(iL) for iL in load_ids]
    E = np.zeros((D, len(load_ids)), dtype=real_dtype)
    for local_id, iL in enumerate(load_ids):
        E[iL, local_id] = 1
    print(('macroscopic load batch E ids = ' + str(load_ids)))
    EN = Tensor(name='EN_batch', N=Nbar, shape=(D, len(load_ids)), Fourier=False, fft_form=fft_form, dtype=real_dtype)
    EN.set_mean(E)
    x0 = EN.zeros_like(name='x0')
    profile_enabled = _profile_enabled(pb)
    if profile_enabled:
        raise NotImplementedError("load_batch_size>1 no esta implementado con profile_timing=True.")
    Afun_local = Afun
    rhs_t0 = time.perf_counter()
    B = Afun_local(-EN)
    cupy_synchronize(B)
    rhs_build_s = time.perf_counter() - rhs_t0
    cb = _make_solver_callback(
        _solver_callback_mode(pb),
        Afun=Afun_local,
        B=B,
        EN=EN,
        GN=GN,
        A_Ga=A,
    )
    par = dict(pb.solver)
    par['batched_rhs'] = True
    load_t0 = time.perf_counter()
    X, info = linear_solver(solver=pb.solver['kind'], Afun=Afun_local, B=B, x0=x0, par=par, callback=cb)
    load_wall_s = time.perf_counter() - load_t0
    X = add_macro2minimizer(
        X,
        E,
        check_mean=_check_macro_mean_enabled(pb),
        inplace=_fast_macro_add_enabled(pb),
    )
    batch_solutions = _split_batched_solution(X, load_ids)
    result_template = {
        'cb': cb,
        'info': info,
        'batched_load_ids': list(load_ids),
        'rhs_build_s': float(rhs_build_s),
        'load_wall_s': float(load_wall_s),
    }
    return load_ids, batch_solutions, [dict(result_template) for _ in load_ids]


def solve_affine_sensitivity_fields(D, Nbar, Afun, pb, GN, solutions, fft_form='c'):
    """Solve exact discrete affine sensitivities around the current primal anchor.

    The primal corrector satisfies ``G A(gamma) e = 0``.  Differentiating with
    respect to an affine coefficient ``gamma_q`` gives
    ``G A(gamma) de_q = -G dA_q e`` with zero macroscopic mean.
    """
    cfields = pb.solve.get('affine_sensitivity_cfields', None)
    if cfields is None:
        return None, {}

    names = list(pb.solve.get('affine_sensitivity_names', []))
    q_count = int(len(cfields))
    if not names:
        names = ['gamma_{0}'.format(q) for q in range(q_count)]
    if len(names) != q_count:
        raise ValueError("affine_sensitivity_names no coincide con affine_sensitivity_cfields.")

    active_load_ids = pb.solve.get('active_load_ids', None)
    if active_load_ids is None:
        load_ids = list(range(D))
    else:
        load_ids = sorted({int(value) for value in active_load_ids})
    if any(not hasattr(solutions[iL], 'val') for iL in load_ids):
        raise RuntimeError("No hay soluciones primales para calcular sensibilidades afines.")

    batch_size = max(
        1,
        int(pb.solve.get('affine_sensitivity_batch_size', pb.solve.get('load_batch_size', 1))),
    )
    batch_size = min(batch_size, len(load_ids))
    real_dtype = _solver_real_dtype(pb)
    par = dict(pb.solver)

    field_shape = tuple(int(v) for v in np.asarray(Nbar, dtype=int).tolist())
    consumer = pb.solve.get('affine_sensitivity_consumer', None)
    streaming = callable(consumer)
    fields = None if streaming else np.empty(
        (q_count, len(load_ids), D) + field_shape,
        dtype=np.dtype(real_dtype),
    )
    solver_infos = []
    consumed = []
    host_transfer_wall_s = 0.0
    consumer_wall_s = 0.0

    def make_sensitivity_tensor(q, spec):
        if isinstance(spec, dict):
            storage = str(spec.get('storage', spec.get('multype', ''))).strip().lower()
            if storage in {'sym21_indexed', 'indexed_sym21'}:
                index = spec.get('index', None)
                table = spec.get('table', None)
                if index is None or table is None:
                    raise ValueError("Descriptor sym21_indexed sin index/table.")
                index_shape = tuple(int(value) for value in getattr(index, 'shape', ()))
                if index_shape == field_shape:
                    val = index[None, ...]
                elif index_shape == (1,) + field_shape:
                    val = index
                else:
                    raise ValueError(
                        "Indice dA/dgamma incompatible: "
                        "se esperaba (*N) o (1, *N)."
                    )
                tensor = Tensor(
                    name='dA_dgamma_{0}'.format(q),
                    val=val,
                    order=1,
                    N=Nbar,
                    Y=pb.Y,
                    multype='sym21_indexed',
                    Fourier=False,
                    origin=0,
                    fft_form=fft_form,
                )
                tensor.material_table = table
                return tensor
            if 'cfield' in spec:
                spec = spec['cfield']
            else:
                raise ValueError("Descriptor dA/dgamma no soportado: {0}".format(storage))

        cfield_shape = tuple(int(value) for value in np.shape(spec))
        if cfield_shape[:1] == (21,):
            multype = 'sym21'
            order = 1
        elif cfield_shape[:2] == (D, D):
            multype = 21
            order = 2
        else:
            raise ValueError(
                "Campo dA/dgamma incompatible: "
                "se esperaba (21, *N) o (6, 6, *N)."
            )
        return Tensor(
            name='dA_dgamma_{0}'.format(q),
            val=spec,
            order=order,
            N=Nbar,
            Y=pb.Y,
            multype=multype,
            Fourier=False,
            origin=0,
            fft_form=fft_form,
        )

    def emit_or_store(q, storage_index, load_id, value):
        nonlocal host_transfer_wall_s, consumer_wall_s
        transfer_started = time.perf_counter()
        host_value = to_host_array(value).astype(np.dtype(real_dtype), copy=False)
        host_transfer_wall_s += float(time.perf_counter() - transfer_started)
        if streaming:
            consumer_started = time.perf_counter()
            consumer(int(q), str(names[q]), int(load_id), host_value)
            consumer_wall_s += float(time.perf_counter() - consumer_started)
            consumed.append((int(q), int(load_id)))
        else:
            fields[int(q), int(storage_index)] = host_value

    for q, cfield in enumerate(cfields):
        dA = make_sensitivity_tensor(q, cfield)

        for start in range(0, len(load_ids), batch_size):
            chunk_load_ids = load_ids[start:start + batch_size]
            if len(chunk_load_ids) == 1:
                load_id = int(chunk_load_ids[0])
                stress_rhs = dA(solutions[load_id])
                B = -(GN(stress_rhs))
                x0 = B.zeros_like(name='x0_sens')
                X, info = linear_solver(
                    solver=pb.solver['kind'],
                    Afun=Afun,
                    B=B,
                    x0=x0,
                    par=par,
                    callback=None,
                )
                emit_or_store(q, start, load_id, X.val)
                solver_infos.append({
                    'coefficient_index': int(q),
                    'coefficient_name': str(names[q]),
                    'load_ids': [load_id],
                    'info': info,
                })
                continue

            solution_values = [solutions[iL].val for iL in chunk_load_ids]
            xp = get_array_module(dA.val, *solution_values)
            if xp is np:
                val = np.stack([to_host_array(value) for value in solution_values], axis=1)
            else:
                val = xp.stack(
                    [to_backend_array(value, prefer_backend='cupy') for value in solution_values],
                    axis=1,
                )
            sol_batch = solutions[int(chunk_load_ids[0])].copy(
                name='sol_total_batch',
                val=val,
                order=2,
            )
            stress_rhs = dA(sol_batch)
            B = -(GN(stress_rhs))
            x0 = B.zeros_like(name='x0_sens_batch')
            batch_par = dict(par)
            batch_par['batched_rhs'] = True
            X, info = linear_solver(
                solver=pb.solver['kind'],
                Afun=Afun,
                B=B,
                x0=x0,
                par=batch_par,
                callback=None,
            )
            x_host = to_host_array(X.val).astype(np.dtype(real_dtype), copy=False)
            for local, load_id in enumerate(chunk_load_ids):
                emit_or_store(q, start + local, load_id, x_host[:, local])
            solver_infos.append({
                'coefficient_index': int(q),
                'coefficient_name': str(names[q]),
                'load_ids': [int(value) for value in chunk_load_ids],
                'info': info,
            })

    rel_residuals = []
    convergence_flags = []
    iterations = []
    for item in solver_infos:
        info = item.get('info', {})
        if 'norm_res_rel_per_rhs' in info:
            rel_residuals.extend(float(value) for value in info['norm_res_rel_per_rhs'])
        elif 'norm_res_rel' in info:
            rel_residuals.append(float(info['norm_res_rel']))
        elif 'norm_res' in info and 'rhs_norm' in info:
            rhs_norm = max(float(info['rhs_norm']), np.finfo(float).tiny)
            rel_residuals.append(float(info['norm_res']) / rhs_norm)
        if 'converged_per_rhs' in info:
            convergence_flags.extend(bool(value) for value in info['converged_per_rhs'])
        elif 'converged' in info:
            convergence_flags.append(bool(info['converged']))
        if 'kit' in info:
            iterations.append(int(info['kit']))

    summary = {
        'coefficient_names': names,
        'load_ids': [int(value) for value in load_ids],
        'batch_size': int(batch_size),
        'solve_count': int(len(solver_infos)),
        'rhs_count': int(q_count * len(load_ids)),
        'all_converged': bool(all(convergence_flags)) if convergence_flags else False,
        'final_norm_res_rel_max': float(max(rel_residuals)) if rel_residuals else np.nan,
        'cg_iterations_max': int(max(iterations)) if iterations else -1,
        'streamed': bool(streaming),
        'host_transfer_wall_s': float(host_transfer_wall_s),
        'consumer_wall_s': float(consumer_wall_s),
    }
    output = {
        'coefficient_names': names,
        'load_ids': [int(value) for value in load_ids],
        'field_shape': [int(D)] + [int(value) for value in field_shape],
        'streamed': bool(streaming),
    }
    if streaming:
        output['consumed'] = consumed
    else:
        output['fields'] = fields
    return output, summary


def stream_primal_solution_fields_before_sensitivity(D, Nbar, pb, solutions, load_ids):
    consumer = pb.solve.get('solution_field_consumer', None)
    if not callable(consumer):
        return
    field_dtype = np.dtype(pb.solve.get('solution_field_dtype', _solver_real_dtype(pb)))
    field_shape = tuple(int(v) for v in np.asarray(Nbar, dtype=int).tolist())
    consumed = []
    for load_id in load_ids:
        load_id = int(load_id)
        if not hasattr(solutions[load_id], 'val'):
            raise RuntimeError("No hay solucion primal para emitir campos en streaming.")
        total = to_host_array(solutions[load_id].val).astype(field_dtype, copy=False)
        fluctuation = total.copy()
        fluctuation[load_id] -= field_dtype.type(1.0)
        consumer(load_id, fluctuation)
        consumed.append(load_id)
    pb.output['solution_fields_primal'] = {
        'load_ids': consumed,
        'field_shape': [int(D)] + [int(value) for value in field_shape],
        'dtype': str(field_dtype),
        'streamed': True,
    }


def _is_free_threaded_python():
    checker = getattr(sys, '_is_gil_enabled', None)
    if checker is not None:
        try:
            return not bool(checker())
        except TypeError:
            pass
    return bool(sysconfig.get_config_var('Py_GIL_DISABLED'))


def _resolve_parallel_backend(pb, n_tasks):
    backend = pb.solve.get('parallel_backend', 'auto')
    if backend not in {'auto', 'process', 'thread'}:
        raise ValueError("parallel_backend debe ser 'auto', 'process' o 'thread'.")

    workers = max(1, min(int(pb.solve.get('parallel_workers', n_tasks)), n_tasks))
    if backend == 'auto':
        backend = 'thread' if _is_free_threaded_python() else 'process'

    if backend == 'process':
        override_reasons = []
        if _profile_enabled(pb):
            override_reasons.append(
                "profile_timing envuelve operadores en TimedCallable y no es seguro con ProcessPool"
            )
        if pb.solve.get('fft_backend') == 'cupy':
            override_reasons.append(
                "cupy no debe paralelizar load-cases con procesos independientes"
            )
        if override_reasons:
            print("[parallel] overriding process -> thread: {}".format("; ".join(override_reasons)))
            backend = 'thread'

    return backend, workers


def _parallel_executor(backend, workers):
    if backend == 'thread':
        return ThreadPoolExecutor(max_workers=workers)
    return ProcessPoolExecutor(max_workers=workers)


def scalar(problem):
    """
    Homogenization of scalar elliptic problem.

    Parameters
    ----------
    problem : object
    """
    print(' ')
    pb = problem
    print(pb)

    # Fourier projections
    fft_form = pb.solve.get('fft_form', 'c')
    real_dtype = _solver_real_dtype(pb)
    _, hG1N, hG2N = proj.scalar(
        pb.solve['N'],
        pb.Y,
        NyqNul=True,
        tensor=True,
        dtype=real_dtype,
    )

    if pb.solve['kind'] == 'GaNi':
        Nbar = pb.solve['N']
    elif pb.solve['kind'] == 'Ga':
        Nbar = 2*pb.solve['N'] - 1
        hG1N = hG1N.enlarge(Nbar)
        hG2N = hG2N.enlarge(Nbar)

    FN = DFT(name='FN', inverse=False, N=Nbar, fft_form=fft_form)
    FiN = DFT(name='FiN', inverse=True, N=Nbar, fft_form=fft_form)

    G1N = Operator(name='G1', mat=[[FiN, hG1N, FN]])
    G2N = Operator(name='G2', mat=[[FiN, hG2N, FN]])

    for primaldual in pb.solve['primaldual']:
        tim = Timer(name='primal-dual')
        print(('\nproblem: ' + primaldual))
        solutions = np.zeros(pb.shape).tolist()
        results = np.zeros(pb.shape).tolist()
        load_timings = []
        load_solver_infos = []

        # material coefficients
        mat = Material(pb.material)

        if pb.solve['kind'] == 'GaNi':
            A = mat.get_A_GaNi(pb.solve['N'], primaldual, fft_form=fft_form)
        elif pb.solve['kind'] == 'Ga':
            A = mat.get_A_Ga(Nbar=Nbar, primaldual=primaldual)

        if primaldual == 'primal':
            GN = G1N
        else:
            GN = G2N

        Afun = Operator(name='FiGFA', mat=[[GN, A]])

        # ----------------------------------------------------------------
        # PARALELIZACIÓN POR CASOS DE CARGA (Load Cases)
        # ----------------------------------------------------------------
        parallel = pb.solve.get('parallel', True)
        parallel_backend = 'serial'
        parallel_workers = 1
        if parallel:
            parallel_backend, parallel_workers = _resolve_parallel_backend(pb, pb.dim)
            print('load-case backend = {} (workers={})'.format(parallel_backend, parallel_workers))
            with _parallel_executor(parallel_backend, parallel_workers) as executor:
                futures = [executor.submit(solve_load_scalar, iL, pb.dim, Nbar, Afun, pb, GN, A,
                                           add_macro2minimizer, linear_solver, CallBack, CallBack_GA, fft_form)
                           for iL in np.arange(pb.dim)]
                for future in futures:
                    iL, sol, res = future.result()
                    solutions[iL] = sol
                    results[iL] = res
                    if _profile_enabled(pb) and 'timing' in res:
                        load_timings.append(res['timing'])
        else:
            for iL in np.arange(pb.dim):
                iL, sol, res = solve_load_scalar(iL, pb.dim, Nbar, Afun, pb, GN, A,
                                                 add_macro2minimizer, linear_solver, CallBack, CallBack_GA, fft_form)
                solutions[iL] = sol
                results[iL] = res
                if _profile_enabled(pb) and 'timing' in res:
                    load_timings.append(res['timing'])
        tim.measure()
        if _profile_enabled(pb):
            profile_summary = _aggregate_load_timings(
                load_timings, primaldual, parallel, parallel_backend, parallel_workers, tim.vals[-1][1]
            )
            pb.output.setdefault('solver_timing', {})[primaldual] = profile_summary
            _print_timing_summary(profile_summary)

        # POSTPROCESSING
        del Afun, GN
        postprocess(pb, A, mat, solutions, results, primaldual)


def elasticity(problem):
    """
    Homogenization of linear elasticity.

    Parameters
    ----------
    problem : object
    """
    print(' ')
    pb = problem
    print(pb)

    # Fourier projections
    projection_t0 = _stage_tic()
    fft_form = pb.solve.get('fft_form', 'c')
    real_dtype = _solver_real_dtype(pb)
    projection_storage = _projection_storage(pb)
    projection_backend = _projection_backend(pb)
    if pb.solve.get('kind') != 'GaNi' and projection_backend in {'auto', 'cupy'}:
        projection_backend = 'numpy'
    projection_cache_enabled = bool(pb.solve.get('cache_projection', False))
    projection_cache_key = _elasticity_projection_cache_key(
        pb,
        fft_form=fft_form,
        real_dtype=real_dtype,
        projection_storage=projection_storage,
        projection_backend=projection_backend,
    )
    cached_projection = (
        _ELASTICITY_PROJECTION_CACHE.get(projection_cache_key)
        if projection_cache_enabled and pb.solve.get('kind') == 'GaNi'
        else None
    )
    projection_cache_hit = cached_projection is not None
    if projection_cache_hit:
        hG1N, hG2N = cached_projection
        hG1hN = hG1sN = hG2hN = hG2sN = None
    else:
        if projection_storage in {'sym21', 'symmetric21', 'packed21', 'direct', 'formula', 'elastic_direct'}:
            hG1N, hG2N = proj.elasticity_combined(
                pb.solve['N'],
                pb.Y,
                NyqNul=True,
                tensor=True,
                fft_form=fft_form,
                dtype=real_dtype,
                storage=projection_storage,
                backend=projection_backend,
            )
            hG1hN = hG1sN = hG2hN = hG2sN = None
        else:
            _, hG1hN, hG1sN, hG2hN, hG2sN = proj.elasticity(
                pb.solve['N'],
                pb.Y,
                NyqNul=True,
                tensor=True,
                fft_form=fft_form,
                dtype=real_dtype,
            )
            hG1N = hG2N = None

    if pb.solve['kind'] == 'GaNi':
        Nbar = pb.solve['N']
    elif pb.solve['kind'] == 'Ga':
        Nbar = 2*pb.solve['N'] - 1
        if hG1N is not None:
            hG1N = hG1N.enlarge(Nbar)
            hG2N = hG2N.enlarge(Nbar)
        else:
            hG1hN = hG1hN.enlarge(Nbar)
            hG1sN = hG1sN.enlarge(Nbar)
            hG2hN = hG2hN.enlarge(Nbar)
            hG2sN = hG2sN.enlarge(Nbar)

    FN = DFT(name='FN', inverse=False, N=Nbar, fft_form=fft_form)
    FiN = DFT(name='FiN', inverse=True, N=Nbar, fft_form=fft_form)

    if hG1N is None:
        hG1N = hG1hN + hG1sN
        hG2N = hG2hN + hG2sN

    if (
        projection_cache_enabled
        and not projection_cache_hit
        and pb.solve.get('kind') == 'GaNi'
    ):
        _ELASTICITY_PROJECTION_CACHE[projection_cache_key] = (hG1N, hG2N)

    G1N = Operator(name='G1', mat=[[FiN, hG1N, FN]])
    G2N = Operator(name='G2', mat=[[FiN, hG2N, FN]])
    projection_s = _stage_toc(projection_t0)
    app_timing = pb.output.setdefault('application_timing', {})
    app_timing['projection_s'] = float(projection_s)
    app_timing['projection_storage'] = str(projection_storage)
    app_timing['projection_backend'] = str(projection_backend)
    app_timing['fft_form'] = str(fft_form)
    app_timing['projection_cache_enabled'] = bool(projection_cache_enabled)
    app_timing['projection_cache_hit'] = bool(projection_cache_hit)

    for primaldual in pb.solve['primaldual']:
        primaldual_t0 = _stage_tic()
        tim = Timer(name='primal-dual')
        print(('\nproblem: ' + primaldual))
        solutions = np.zeros(pb.shape).tolist()
        results = np.zeros(pb.shape).tolist()
        load_timings = []
        load_solver_infos = []

        # material coefficients
        material_t0 = _stage_tic()
        mat = Material(pb.material)

        if pb.solve['kind'] == 'GaNi':
            A = mat.get_A_GaNi(pb.solve['N'], primaldual, fft_form=fft_form)
        elif pb.solve['kind'] == 'Ga':
            A = mat.get_A_Ga(Nbar=Nbar, primaldual=primaldual)

        if primaldual == 'primal':
            GN = G1N
        else:
            GN = G2N

        Afun = Operator(name='FiGFA', mat=[[GN, A]])
        material_operator_s = _stage_toc(material_t0)

        # ----------------------------------------------------------------
        # PARALELIZACIÓN POR CASOS DE CARGA (Load Cases)
        # ----------------------------------------------------------------
        solve_t0 = _stage_tic()
        D = int(pb.dim*(pb.dim+1)/2)
        active_load_ids = pb.solve.get('active_load_ids', None)
        if active_load_ids is None:
            load_ids_to_solve = list(range(D))
        else:
            load_ids_to_solve = sorted({int(value) for value in active_load_ids})
            if not load_ids_to_solve or any(value < 0 or value >= D for value in load_ids_to_solve):
                raise ValueError("active_load_ids contiene cargas inválidas.")
        parallel = pb.solve.get('parallel', True)
        load_batch_size = _load_batch_size(pb, D)
        if load_batch_size > 1 and parallel:
            print('load batching disables load-case parallelism (batch_size={})'.format(load_batch_size))
            parallel = False
        parallel_backend = 'serial'
        parallel_workers = 1
        if parallel:
            parallel_backend, parallel_workers = _resolve_parallel_backend(
                pb, len(load_ids_to_solve)
            )
            print('load-case backend = {} (workers={})'.format(parallel_backend, parallel_workers))
            with _parallel_executor(parallel_backend, parallel_workers) as executor:
                futures = [executor.submit(solve_load_elasticity, iL, D, Nbar, Afun, pb, GN, A,
                                           add_macro2minimizer, linear_solver, CallBack, CallBack_GA, fft_form)
                           for iL in load_ids_to_solve]
                for future in futures:
                    iL, sol, res = future.result()
                    solutions[iL] = sol
                    results[iL] = res
                    load_solver_infos.append(_solver_info_payload(res.get('info', {}), [iL]))
                    if _profile_enabled(pb) and 'timing' in res:
                        load_timings.append(res['timing'])
        else:
            if load_batch_size > 1 and len(load_ids_to_solve) > 1:
                for start in range(0, len(load_ids_to_solve), load_batch_size):
                    load_ids = load_ids_to_solve[start:start + load_batch_size]
                    batch_ids, batch_solutions, batch_results = solve_load_elasticity_batch(
                        load_ids, D, Nbar, Afun, pb, GN, A,
                        add_macro2minimizer, linear_solver, CallBack, CallBack_GA, fft_form,
                    )
                    if batch_results:
                        load_solver_infos.append(
                            _solver_info_payload(batch_results[0].get('info', {}), batch_ids)
                        )
                    for iL, sol, res in zip(batch_ids, batch_solutions, batch_results):
                        solutions[iL] = sol
                        results[iL] = res
            else:
                for iL in load_ids_to_solve:
                    iL, sol, res = solve_load_elasticity(iL, D, Nbar, Afun, pb, GN, A,
                                                         add_macro2minimizer, linear_solver, CallBack, CallBack_GA, fft_form)
                    solutions[iL] = sol
                    results[iL] = res
                    load_solver_infos.append(_solver_info_payload(res.get('info', {}), [iL]))
                    if _profile_enabled(pb) and 'timing' in res:
                        load_timings.append(res['timing'])
        tim.measure()
        solve_s = _stage_toc(solve_t0)
        if _profile_enabled(pb):
            profile_summary = _aggregate_load_timings(
                load_timings, primaldual, parallel, parallel_backend, parallel_workers, tim.vals[-1][1]
            )
            pb.output.setdefault('solver_timing', {})[primaldual] = profile_summary
            _print_timing_summary(profile_summary)

        if (
            primaldual == 'primal'
            and pb.solve.get('affine_sensitivity_cfields', None) is not None
            and bool(pb.solve.get('solution_field_stream_before_sensitivity', False))
        ):
            stream_primal_solution_fields_before_sensitivity(
                D, Nbar, pb, solutions, load_ids_to_solve
            )

        affine_sensitivity_summary = {}
        if (
            primaldual == 'primal'
            and pb.solve.get('affine_sensitivity_cfields', None) is not None
        ):
            sens_t0 = _stage_tic()
            sensitivity_output, affine_sensitivity_summary = solve_affine_sensitivity_fields(
                D, Nbar, Afun, pb, GN, solutions, fft_form
            )
            affine_sensitivity_summary['wall_s'] = float(_stage_toc(sens_t0))
            affine_sensitivity_summary['solver_wall_excluding_consumer_s'] = float(
                max(
                    0.0,
                    affine_sensitivity_summary['wall_s']
                    - float(affine_sensitivity_summary.get('consumer_wall_s', 0.0)),
                )
            )
            if sensitivity_output is not None:
                pb.output['affine_sensitivity_' + primaldual] = sensitivity_output
                pb.output['affine_sensitivity_summary_' + primaldual] = affine_sensitivity_summary

        # POSTPROCESSING
        partial_load_output = bool(pb.solve.get('partial_load_output', False))
        if partial_load_output:
            partial_columns = {}
            for iL in load_ids_to_solve:
                stress = A(solutions[iL])
                partial_columns[int(iL)] = np.asarray(stress.mean(), dtype=float)
            pb.output['partial_columns_' + primaldual] = partial_columns
            pb.output['res_' + primaldual] = results
            if pb.solve.get('store_solution_fields', False):
                pb.output['sol_' + primaldual] = solutions
            postprocess_s = 0.0
        else:
            postprocess_t0 = _stage_tic()
            postprocess(pb, A, mat, solutions, results, primaldual)
            postprocess_s = _stage_toc(postprocess_t0)
        del Afun, GN
        app_timing.setdefault('primaldual', {})[primaldual] = {
            'material_operator_s': float(material_operator_s),
            'load_solve_s': float(solve_s),
            'postprocess_s': float(postprocess_s),
            'total_primaldual_s': float(_stage_toc(primaldual_t0)),
            'load_batch_size': int(load_batch_size),
            'parallel': bool(parallel),
            'parallel_backend': str(parallel_backend),
            'parallel_workers': int(parallel_workers),
            'load_solver_infos': load_solver_infos,
            'load_solver_summary': _summarize_solver_infos(
                load_solver_infos,
                rtol=pb.solver.get('rtol', pb.solver.get('tol', np.nan)),
                atol=pb.solver.get('atol', 0.0),
                maxiter=pb.solver.get('maxiter', 0),
            ),
            'affine_sensitivity_summary': affine_sensitivity_summary,
        }


if __name__ == '__main__':
    # Este bloque solo se usa para tests directos del módulo ffthompy.applications
    pass
