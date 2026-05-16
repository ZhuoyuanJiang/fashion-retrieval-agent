"""Build a self-contained interactive HTML listening demo for the TTS pilot.

Embeds every audio_out/<model>/<tag>.wav (and the shared reference clip) as a
base64 data URI, so the resulting demo.html is one portable file — open it
anywhere, no sibling .wav files needed. The page also documents how the pilot
data was built: the captions sampled from FACap, the shared reference voice,
the candidate models, and the WER scoring.
"""
import json, base64, html, pathlib

ROOT = pathlib.Path("/tmp3/zhuoyuan/tts_pilot")
AUDIO = ROOT / "audio_out"
REF = ROOT / "refs" / "ref_en.wav"
OUT = pathlib.Path.home() / "tts_pilot_demo" / "demo.html"

caps = json.load(open(ROOT / "sample_captions.json"))
try:
    wer = json.load(open(ROOT / "wer_results.json"))
except FileNotFoundError:
    wer = {}

# display name, param size, origin — keyed by the audio_out/<dir> name
META = {
    "f5tts":      ("F5-TTS",         "~0.33B", "SWivid"),
    "chatterbox": ("Chatterbox",     "~0.5B",  "Resemble AI"),
    "higgs":      ("Higgs Audio V2", "~5.8B",  "Boson AI"),
}
ORDER = ["f5tts", "chatterbox", "higgs"]

# why each caption was sampled
ROLE = {
    "shortest":        "Shortest caption in the FACap dress slice — the floor of the length range.",
    "longest":         "Longest caption — a multi-clause stress test for length and coherence.",
    "fashion-vocab-1": "Fashion-vocabulary dense — \"pleated bodice\", \"ruffled lace trim\".",
    "fashion-vocab-2": "Fashion-vocabulary dense — \"V-neckline\", \"sheer fabric\", \"lace-trimmed hem\".",
    "fashion-vocab-3": "Fashion-vocabulary dense — \"sweetheart neckline\", \"satin-like fabric\".",
}

present = {d.name for d in AUDIO.iterdir() if d.is_dir() and any(d.iterdir())}
models = [m for m in ORDER if m in present] + sorted(present - set(ORDER))


def b64(p: pathlib.Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def wer_badge(frac):
    if frac is None:
        return '<span class="badge gray">not scored</span>'
    pct = frac * 100
    cls = "green" if pct < 2 else ("amber" if pct < 10 else "red")
    return f'<span class="badge {cls}">WER {pct:.1f}%</span>'


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
       Helvetica, Arial, sans-serif; background: #f5f6f8; color: #1d2127;
       line-height: 1.62; }
.wrap { max-width: 980px; margin: 0 auto; padding: 44px 24px 96px; }
header h1 { font-size: 28px; letter-spacing: -0.3px; }
.sub { color: #6b7280; font-size: 15px; margin-top: 4px; }
section { margin-top: 40px; }
h2 { font-size: 19px; margin-bottom: 14px; padding-bottom: 7px;
     border-bottom: 2px solid #e4e7eb; }
h3 { font-size: 14px; margin: 16px 0 3px; color: #2563eb;
     text-transform: uppercase; letter-spacing: 0.5px; }
p { margin-bottom: 9px; }
a { color: #2563eb; }
.card { background: #fff; border: 1px solid #e4e7eb; border-radius: 11px;
        padding: 20px 22px; margin-bottom: 16px; }
.note { background: #fffbeb; border: 1px solid #fde68a; border-radius: 9px;
        padding: 13px 16px; font-size: 14px; margin: 12px 0; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
       background: #eef0f3; padding: 1px 5px; border-radius: 4px; font-size: 13px; }
table { border-collapse: collapse; width: 100%; margin-top: 6px; font-size: 14px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #e4e7eb; }
th { background: #f0f2f5; font-weight: 600; }
.tag { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
       font-size: 12px; font-weight: 600; color: #fff; background: #2563eb;
       padding: 3px 9px; border-radius: 5px; }
.len { color: #6b7280; font-size: 13px; margin-left: 8px; }
.caption-text { font-size: 16px; background: #f0f4ff; border-left: 3px solid #2563eb;
       padding: 11px 15px; border-radius: 6px; margin: 11px 0 6px; }
.role { color: #6b7280; font-size: 13px; font-style: italic; }
.models { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
          margin-top: 16px; }
@media (max-width: 740px) { .models { grid-template-columns: 1fr; } }
.cell { background: #fafbfc; border: 1px solid #e4e7eb; border-radius: 9px;
        padding: 13px 14px; }
.cell .mname { font-weight: 600; font-size: 15px; }
.cell .msize { color: #6b7280; font-size: 12px; margin-left: 6px; }
audio { width: 100%; margin: 9px 0 7px; height: 36px; }
.badge { font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 11px; }
.badge.green { background: #dcfce7; color: #15803d; }
.badge.amber { background: #fef3c7; color: #b45309; }
.badge.red   { background: #fee2e2; color: #b91c1c; }
.badge.gray  { background: #e5e7eb; color: #6b7280; }
details { margin-top: 8px; font-size: 13px; }
summary { cursor: pointer; color: #2563eb; }
details p { margin: 6px 0 0; color: #374151; }
.ref-box { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.ref-box audio { width: 320px; }
footer { margin-top: 48px; color: #6b7280; font-size: 13px;
         border-top: 1px solid #e4e7eb; padding-top: 16px; }
"""


def esc(s):
    return html.escape(str(s))


H = ['<!doctype html>', '<html lang="en">', '<head>',
     '<meta charset="utf-8">',
     '<meta name="viewport" content="width=device-width, initial-scale=1">',
     '<title>TTS Pilot — Listening Demo</title>',
     f'<style>{CSS}</style>', '</head>', '<body>', '<div class="wrap">']

# ---- header ----
H += ['<header>',
      '<h1>TTS Pilot — Listening Demo</h1>',
      '<p class="sub">Picking the text-to-speech engine for the audio-native '
      'query tower (Plan 14, fashion-retrieval-agent)</p>',
      '</header>']

# ---- what this is ----
H += ['<section>', '<h2>What you are listening to</h2>', '<div class="card">',
      '<p>Plan 14 swaps the retrieval model\'s query-side modification input '
      'from typed text to <b>spoken audio</b>. FACap modifications are '
      'text-only, so they must be synthesized to speech. This pilot synthesizes '
      'the same 5 captions with several candidate TTS models so the engine for '
      'the full ~55k-utterance synthesis can be chosen by listening.</p>',
      '<p>All candidates score essentially tied on intelligibility (WER ~0.9%), '
      'so the words are not the deciding factor — <b>judge on voice naturalness, '
      'prosody, and how cleanly each handles fashion vocabulary.</b></p>',
      '</div>', '</section>']

# ---- how it was built ----
H += ['<section>', '<h2>How this demo was constructed</h2>', '<div class="card">']

H += ['<h3>1 · The captions</h3>',
      '<p>The 5 captions are real <b>modification strings from the FACap '
      'dataset, dress slice</b> — the natural-language edits ("change dress A '
      'into dress B") that are the query-side input to the two-tower retrieval '
      'model (~55k of them, text-only, median ~90 characters). They were '
      'sampled to span the range the full synthesis will hit:</p>',
      '<table><tr><th>Tag</th><th>Length</th><th>Why this one</th></tr>']
for tag in caps:
    H.append(f'<tr><td><span class="tag">{esc(tag)}</span></td>'
             f'<td>{len(caps[tag])} chars</td>'
             f'<td>{esc(ROLE.get(tag, ""))}</td></tr>')
H += ['</table>']

H += ['<h3>2 · The voice</h3>',
      '<p>Every model here is a <b>zero-shot voice-cloning TTS</b>: given one '
      'short reference clip, it clones that timbre. For this pilot a <b>single '
      'shared reference clip</b> was used for all models and all captions, so '
      'any difference you hear is the <i>model</i>, not the voice choice. The '
      'reference (F5-TTS\'s bundled English clip, 24&nbsp;kHz mono):</p>',
      '<div class="ref-box">']
if REF.exists():
    H.append(f'<audio controls preload="none" '
             f'src="data:audio/wav;base64,{b64(REF)}"></audio>')
H += ['<span class="role">"Some call me nature, others call me mother '
      'nature."</span>', '</div>',
      '<div class="note"><b>Pilot only.</b> This single voice is just for '
      'engine selection. The real Plan-14 training set will <i>not</i> use one '
      'voice — it draws <b>~110 distinct speakers from VCTK</b> '
      '(gender-balanced, multiple English accents, ~10 held out for a separate '
      'out-of-distribution eval) so the audio query tower learns content '
      'speaker-invariantly.</div>']

H += ['<h3>3 · The models</h3>',
      '<p>Each caption was synthesized with the candidates below, all '
      'zero-shot voice-cloned from the clip above, all run locally on a single '
      'GPU. Audio model sizes are small next to a 7B LLM — except Higgs, the '
      '"bigger model" check:</p>',
      '<table><tr><th>Model</th><th>Params</th><th>Origin</th></tr>']
for m in models:
    name, size, origin = META.get(m, (m, "?", "?"))
    H.append(f'<tr><td><b>{esc(name)}</b></td><td>{esc(size)}</td>'
             f'<td>{esc(origin)}</td></tr>')
H += ['</table>']

H += ['<h3>4 · WER scoring</h3>',
      '<p>After synthesis, <code>faster-whisper medium.en</code> transcribes '
      'each clip and <code>jiwer</code> computes word error rate against the '
      'normalized source caption (lower-cased, punctuation stripped). Low WER '
      '= the words survived synthesis = the modification\'s meaning is '
      'recoverable, which is what retrieval training needs. Expand "what '
      'Whisper heard" under any clip to see the transcript.</p>',
      '</div>', '</section>']

# ---- WER summary ----
H += ['<section>', '<h2>WER summary</h2>', '<div class="card">',
      '<table><tr><th>Model</th><th>avg WER (lower = better)</th></tr>']
for m in models:
    aw = wer.get(m, {}).get("avg_wer")
    H.append(f'<tr><td><b>{esc(META.get(m, (m,))[0])}</b></td>'
             f'<td>{wer_badge(aw)}</td></tr>')
H += ['</table></div>', '</section>']

# ---- the comparison ----
H += ['<section>', '<h2>Listen &amp; compare</h2>',
      '<p style="color:#6b7280;font-size:14px">One clip plays at a time — '
      'starting a clip pauses the others, so A/B comparison is just clicking '
      'across a row.</p>']
for tag, text in caps.items():
    H += ['<div class="card">',
          f'<div><span class="tag">{esc(tag)}</span>'
          f'<span class="len">{len(text)} chars</span></div>',
          f'<div class="caption-text">{esc(text)}</div>',
          f'<div class="role">{esc(ROLE.get(tag, ""))}</div>',
          '<div class="models">']
    for m in models:
        name, size, _ = META.get(m, (m, "?", "?"))
        src = AUDIO / m / f"{tag}.wav"
        cell = [f'<div class="cell">',
                f'<div><span class="mname">{esc(name)}</span>'
                f'<span class="msize">{esc(size)}</span></div>']
        if src.exists():
            cell.append(f'<audio controls preload="none" '
                        f'src="data:audio/wav;base64,{b64(src)}"></audio>')
        else:
            cell.append('<p class="role">no audio</p>')
        pc = wer.get(m, {}).get("per_caption", {}).get(tag, {})
        cell.append(wer_badge(pc.get("wer")))
        if pc.get("hyp"):
            cell.append('<details><summary>what Whisper heard</summary>'
                        f'<p>{esc(pc["hyp"])}</p></details>')
        cell.append('</div>')
        H.append("".join(cell))
    H += ['</div>', '</div>']
H += ['</section>']

# ---- other candidates ----
H += ['<section>', '<h2>Candidates not in this demo</h2>', '<div class="card">',
      '<p><b>IndexTTS-2</b> — not run: its dependencies pin torch 2.8 / CUDA '
      '12.8, and this server\'s driver is CUDA 12.4. Would need a newer-driver '
      'host.</p>',
      '<p><b>Higgs Audio V2</b> <i>is</i> included above — it took pinning both '
      'the checkpoint and audio-tokenizer repos to their last pre-2026 '
      'revisions, because the current versions were migrated to a native-'
      '<code>transformers</code> format the standalone serving code cannot '
      'load.</p>',
      '</div>', '</section>']

# ---- footer ----
H += ['<footer>',
      'Pick a winner by listening, then it becomes the synthesis engine for '
      'Plan 14 M2 — the full FACap dress-slice synthesis (~55k utterances, '
      'VCTK-cloned voices). &nbsp;·&nbsp; 5 captions × '
      f'{len(models)} models, single shared reference voice.',
      '</footer>']

H += ['</div>',
      # one-clip-at-a-time playback
      '<script>document.addEventListener("play",function(e){'
      'document.querySelectorAll("audio").forEach(function(a){'
      'if(a!==e.target)a.pause();});},true);</script>',
      '</body>', '</html>']

OUT.write_text("\n".join(H))
size_mb = OUT.stat().st_size / 1e6
print(f"wrote {OUT}  ({size_mb:.1f} MB, self-contained)")
print(f"models: {models}")
print(f"captions: {list(caps)}")
