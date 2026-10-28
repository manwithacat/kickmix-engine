"""Run a command; sample its whole process tree every --interval s: summed RSS and summed phys_footprint (macOS
proc_pid_rusage, as Activity Monitor's "Memory"; same method as the fork's kickmix_tools). Reports the peak of the
sums, wall time and the command's stdout. Usage: memwatch.py OUT.json [--interval 0.05] -- cmd args..."""
import ctypes, ctypes.util, json, os, subprocess, sys, threading, time

def main():
    i = sys.argv.index('--')
    opts, cmd = sys.argv[1:i], sys.argv[i + 1:]
    out_path = opts[0]
    interval = float(opts[opts.index('--interval') + 1]) if '--interval' in opts else 0.05
    if os.path.exists(out_path):
        sys.exit(f'refusing to overwrite {out_path}')
    lib = ctypes.CDLL(ctypes.util.find_library('proc') or '/usr/lib/libproc.dylib', use_errno=True)
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.proc_listchildpids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    def footprint(pid):
        buf = ctypes.create_string_buffer(512)
        if lib.proc_pid_rusage(pid, 2, buf) != 0:
            return None, None
        r = buf.raw
        # rusage_info_v2: uuid[16], user, system, pkg_idle_wkups, interrupt_wkups, pageins, wired_size, resident_size,
        # phys_footprint, ...
        return int.from_bytes(r[64:72], 'little'), int.from_bytes(r[72:80], 'little')

    def children(pid):
        arr = (ctypes.c_int * 4096)()
        n = lib.proc_listchildpids(pid, arr, ctypes.sizeof(arr))
        return list(arr[:max(n, 0)])

    def tree(root):
        seen, todo = [], [root]
        while todo:
            p = todo.pop()
            seen.append(p)
            todo.extend(children(p))
        return seen

    uptime = subprocess.run(['uptime'], capture_output=True, text=True).stdout.strip()
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = dict(rss=0, footprint=0, nproc=0, t_rss=0.0, t_fp=0.0)
    halt = threading.Event()

    def sample():
        while not halt.is_set():
            rs, fs, n = 0, 0, 0
            for p in tree(proc.pid):
                r, f = footprint(p)
                if r is not None:
                    rs += r; fs += f; n += 1
            t = time.perf_counter() - t0
            if rs > peak['rss']:
                peak['rss'], peak['t_rss'] = rs, t
            if fs > peak['footprint']:
                peak['footprint'], peak['t_fp'] = fs, t
            peak['nproc'] = max(peak['nproc'], n)
            halt.wait(interval)

    th = threading.Thread(target=sample, daemon=True); th.start()
    stdout, stderr = proc.communicate()
    wall = time.perf_counter() - t0
    halt.set(); th.join()
    try:
        result = json.loads(stdout)
    except Exception:
        result = None
    rec = dict(cmd=cmd, returncode=proc.returncode, wall_s=wall, peak_rss_mib=peak['rss'] / 2**20,
               peak_footprint_mib=peak['footprint'] / 2**20, t_peak_footprint_s=peak['t_fp'], max_processes=peak['nproc'],
               interval_s=interval, uptime_before=uptime, uptime_after=subprocess.run(['uptime'], capture_output=True, text=True).stdout.strip(),
               result=result, stdout_tail=None if result else stdout[-2000:], stderr_tail=stderr[-2000:])
    json.dump(rec, open(out_path, 'w'), indent=1)
    print(json.dumps({k: rec[k] for k in ('returncode', 'wall_s', 'peak_rss_mib', 'peak_footprint_mib', 'max_processes')}))

main()
