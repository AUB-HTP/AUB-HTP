"""Pre-render the meta-study tables as static HTML, straight from the run's JSON."""
import json, math
from pathlib import Path

ROOT = Path(r"C:\Users\Charbel Hannoun\Documents\AUB\SUMMER 25-26\AUB-HTP")
data = json.loads((ROOT / "pdf_compare" / "results" / "meta_study.json").read_text())

ALPHAS = data["alphas"]
BETAS = data["betas"]
acc = {(r["alpha"], r["beta"]): r for r in data["accuracy"]}

SUP = {"-": "⁻", **{str(i): c for i, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")}}


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f'{m}&times;10<sup>{int(e)}</sup>'


out = []

# ---------------- regime matrix ----------------
rows = ['<thead><tr><th class="rowh">&alpha; \\ &beta;</th>'
        + "".join(f"<th>{b:.2f}</th>" for b in BETAS) + "</tr></thead><tbody>"]
for a in reversed(ALPHAS):
    cells = [f'<th class="rowh">{a:.2f}</th>']
    for b in BETAS:
        r = acc[(a, b)]
        p, f = r["proj_err"], r["fft_err"]
        ratio = max(p, f) / min(p, f)
        proj_wins = p < f
        if ratio < 2.0:
            bg = f"rgba(124,135,148,{0.10 + 0.06*(ratio-1):.3f})"
            fg, label = "#55606c", "tie"
            shown = f"{ratio:.1f}&times;"
        else:
            s = min(math.log10(ratio) / math.log10(650), 1.0)
            al = 0.14 + 0.72 * s
            bg = (f"rgba(45,108,168,{al:.3f})" if proj_wins
                  else f"rgba(177,96,31,{al:.3f})")
            fg = "#ffffff" if s > 0.55 else ("#1d4f7d" if proj_wins else "#7d4315")
            label = "proj" if proj_wins else "fft"
            shown = f"{round(ratio)}&times;"
        title = (f"alpha={a}, beta={b} &mdash; projection {p:.1e}, FFT {f:.1e}")
        cells.append(
            f'<td style="background:{bg};color:{fg}" title="{title}">'
            f'<span class="cell-ratio">{shown}</span>'
            f'<span class="cell-sub">{label}</span></td>')
    rows.append("<tr>" + "".join(cells) + "</tr>")
rows.append("</tbody>")
out.append(("MATRIX", "".join(rows)))

# ---------------- per-alpha means ----------------
means = []
for a in ALPHAS:
    pm = sum(acc[(a, b)]["proj_err"] for b in BETAS) / len(BETAS)
    fm = sum(acc[(a, b)]["fft_err"] for b in BETAS) / len(BETAS)
    means.append((a, pm, fm))
max_f = max(m[2] for m in means)
t = ['<caption>Mean error across &beta;, by &alpha;</caption><thead><tr>'
     '<th class="l">&alpha;</th><th>projection</th><th>inverse FFT</th>'
     '<th class="l">FFT error, to scale</th></tr></thead><tbody>']
for a, pm, fm in means:
    w = max(2, round(230 * fm / max_f))
    t.append(f'<tr><td class="l">{a:.2f}</td>'
             f'<td class="win-proj">{sci(pm)}</td>'
             f'<td class="win-fft">{sci(fm)}</td>'
             f'<td class="l"><span class="bar" style="width:{w}px"></span>'
             f'<span class="barnum">{100*fm:.2f}%</span></td></tr>')
t.append("</tbody>")
out.append(("BYALPHA", "".join(t)))

# ---------------- cost ----------------
NS = ["100", "1000", "10000", "100000"]
t = ['<caption>d=2 &mdash; total time ratio, projection &divide; FFT (build + evaluate)</caption>'
     '<thead><tr><th class="l">&alpha;</th><th>FFT build</th><th>proj build</th>'
     + "".join(f'<th>n={int(n):,}</th>' for n in NS) + "</tr></thead><tbody>"]
for row in data["cost"]:
    cells = [f'<td class="l">{row["alpha"]:.1f}</td>',
             f'<td>{row["fft_build"]:.2f} s</td>',
             f'<td>{row["proj_build"]:.2f} s</td>']
    for n in NS:
        fe, pe = row["per_n"][n]
        ratio = (row["proj_build"] + pe) / (row["fft_build"] + fe)
        cls = "win-proj" if ratio < 1 else "win-fft"
        cells.append(f'<td class="{cls}">{ratio:.2f}</td>')
    t.append("<tr>" + "".join(cells) + "</tr>")
t.append("</tbody>")
out.append(("COST", "".join(t)))

# ---------------- alpha=1 sensitivity ----------------
t = ['<caption>Symmetric law (true &beta; = 0): measured skew, its amplification, '
     'and the resulting error</caption><thead><tr>'
     '<th class="l">&alpha;</th><th>tan(&pi;&alpha;/2)</th><th class="l">sampler</th>'
     '<th>N</th><th>measured &beta;&#770;</th><th>&beta;&#770; &middot; tan</th>'
     '<th>projection</th><th>FFT</th></tr></thead><tbody>']
seen_anti = set()
for r in data["alpha1_sensitivity"]:
    anti = r["antithetic"]
    # antithetic rows are N-independent; show one per alpha
    if anti:
        if r["alpha"] in seen_anti:
            continue
        seen_anti.add(r["alpha"])
    tan = r["tan"]
    tan_s = "&infin;" if tan is None else f"{abs(tan):.2f}"
    amp_s = "&mdash;" if tan is None else f'{r["amplified"]:.4f}'
    n_s = "any" if anti else f'{r["n"]:,}'
    pc = lambda v: f"{100*v:.2f}%"
    pcls = "win-fft" if r["proj_err"] > 0.02 else "win-proj"
    fcls = "win-fft" if r["fft_err"] > 0.02 else "win-proj"
    t.append(f'<tr><td class="l">{r["alpha"]:.2f}</td><td>{tan_s}</td>'
             f'<td class="l">{"antithetic" if anti else "i.i.d."}</td><td>{n_s}</td>'
             f'<td>{r["beta_hat"]:.5f}</td>'
             f'<td class="amp">{amp_s}</td>'
             f'<td class="{pcls}">{pc(r["proj_err"])}</td>'
             f'<td class="{fcls}">{pc(r["fft_err"])}</td></tr>')
t.append("</tbody>")
out.append(("SENS", "".join(t)))

# ---------------- headline numbers ----------------
wins = {"proj": 0, "fft": 0, "tie": 0}
for r in data["accuracy"]:
    wins[r["winner"]] += 1
worst_fft = max(data["accuracy"], key=lambda r: r["fft_err"])
worst_proj = max(data["accuracy"], key=lambda r: r["proj_err"])
max_ratio = max(max(r["proj_err"], r["fft_err"]) / min(r["proj_err"], r["fft_err"])
                for r in data["accuracy"])
print("HEADLINE", json.dumps(dict(
    wins=wins, worst_fft=worst_fft["fft_err"], worst_fft_at=[worst_fft["alpha"], worst_fft["beta"]],
    worst_proj=worst_proj["proj_err"], worst_proj_at=[worst_proj["alpha"], worst_proj["beta"]],
    max_ratio=round(max_ratio), n_spectral=data["n_spectral"])))

dest = Path(__file__).with_name("tables.json")
dest.write_text(json.dumps(dict(out), indent=0), encoding="utf-8")
print("wrote", dest)
