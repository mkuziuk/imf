// Requires pandoc and KaTeX. Override KATEX_MODULE when installed elsewhere.
// All math, fonts and the comparison figure are embedded for offline reading.
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const katexPath = require.resolve(process.env.KATEX_MODULE || '/tmp/imf-report-render/node_modules/katex');
const katex = require(katexPath);
const repo = path.resolve(__dirname, '../..');
const reportDir = path.join(repo, 'tmp/reports');
const input = fs.readFileSync(path.join(__dirname, 'comparison.md'), 'utf8');
const template = fs.readFileSync('/Users/mikhail/.agents/skills/ssh-html-report/assets/dark-report-template.html', 'utf8');
const escape = s => s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const document = JSON.parse(execFileSync('pandoc', ['--from=markdown+tex_math_dollars', '--to=json'], { input, encoding: 'utf8' }));
let displayCount = 0;
let inlineCount = 0;

function math(node, display) {
  display ? displayCount++ : inlineCount++;
  return katex.renderToString(node.c[1], {
    displayMode: display, throwOnError: true, strict: 'error', output: 'htmlAndMathml', trust: false,
  });
}

function transform(node) {
  if (Array.isArray(node)) return node.map(transform);
  if (!node || typeof node !== 'object') return node;
  if (node.t === 'Para' && node.c.length === 1 && node.c[0].t === 'Math' && node.c[0].c[0].t === 'DisplayMath') {
    return { t: 'RawBlock', c: ['html', '<div class="equation">' + math(node.c[0], true) + '</div>'] };
  }
  if (node.t === 'Math') return { t: 'RawInline', c: ['html', math(node, node.c[0].t === 'DisplayMath')] };
  if (node.t === 'Image') {
    const filename = node.c[2][0];
    if (filename.startsWith('/') && fs.existsSync(filename)) {
      node.c[2][0] = 'data:image/png;base64,' + fs.readFileSync(filename).toString('base64');
    }
  }
  if (node.t === 'Link' && node.c[2][0].startsWith('/')) {
    const original = node.c[2][0];
    const match = original.match(/^(.*?)(?::(\d+))?$/);
    const filename = match[1];
    const line = match[2];
    const quantlet = '/Users/mikhail/Projects/IRMF-Intrinsic-Robust-Multiscale-Filtering';
    if (filename === quantlet) {
      node.c[2][0] = 'https://github.com/QuantLet/IRMF-Intrinsic-Robust-Multiscale-Filtering';
    } else if (filename.startsWith(quantlet + '/')) {
      node.c[2][0] = 'https://github.com/QuantLet/IRMF-Intrinsic-Robust-Multiscale-Filtering/blob/58d6762cccbdce3abe5e183ceaf9564bf46768ca/'
        + filename.slice(quantlet.length + 1) + (line ? '#L' + line : '');
    } else if (filename.startsWith(repo + '/') && fs.existsSync(filename)) {
      // Only copy explicitly linked report evidence, not the repository tree.
      const relative = path.relative(repo, filename);
      const destination = path.join(reportDir, 'quantlet-comparison-sources', relative);
      fs.mkdirSync(path.dirname(destination), { recursive: true });
      fs.copyFileSync(filename, destination);
      node.c[2][0] = 'quantlet-comparison-sources/' + relative.split(path.sep).map(encodeURIComponent).join('/');
      node.c[2][1] = original;
    }
  }
  return Object.fromEntries(Object.entries(node).map(([key, value]) => [key, transform(value)]));
}

const body = execFileSync('pandoc', ['--from=json', '--to=html5', '--wrap=none'], {
  input: JSON.stringify(transform(document)), encoding: 'utf8', maxBuffer: 10 * 1024 * 1024,
});
const katexDist = path.dirname(katexPath);
let mathCSS = fs.readFileSync(path.join(katexDist, 'katex.min.css'), 'utf8');
mathCSS = mathCSS.replace(/url\(([^)]+)\)/g, (_, filename) => {
  const extension = path.extname(filename).slice(1);
  const mime = { woff: 'font/woff', woff2: 'font/woff2', ttf: 'font/ttf' }[extension];
  return 'url(data:' + mime + ';base64,' + fs.readFileSync(path.join(katexDist, filename)).toString('base64') + ')';
});
const baseCSS = template.match(/<style>([\s\S]*?)<\/style>/)[1];
const parts = body.split(/(?=<p><strong>)/);
const navigation = [];
const sections = parts.map((part, index) => {
  const id = index === 0 ? 'summary' : 'part-' + index;
  const label = index === 0 ? 'Overview' : part.match(/^<p><strong>(.*?)<\/strong>/)?.[1]?.replace(/\.$/, '') || 'Details';
  navigation.push('<a href="#' + id + '">' + label + '</a>');
  return '<section class="section" id="' + id + '">' + part + '</section>';
}).join('\n');
const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Mathematical and code comparison of QuantLet IRMF and the imf repository.">
<title>QuantLet IRMF and imf: comparison</title>
<style>${baseCSS}\n${mathCSS}
.layout{grid-template-columns:minmax(12rem,16rem) minmax(0,62rem);width:min(100% - 2rem,86rem)}
.section p{max-width:none}.section p:first-child{margin-top:0}.section p:last-child{margin-bottom:0}
.section>p>strong:first-child{color:#edf0f2}
.equation{padding:1rem;overflow-x:auto;overflow-y:hidden}.equation .katex-display{margin:.3em 0}
.katex{font-size:1.12em}.katex-display>.katex{white-space:nowrap}
.table-wrap{margin:1rem 0}table{min-width:34rem}td:first-child{min-width:10rem}
img{display:block;max-width:100%;height:auto;border-radius:.5rem;background:white}
aside{max-height:calc(100vh - 3rem);overflow-y:auto}nav a{font-size:.82rem;line-height:1.4}
.hero h1{max-width:none;font-size:clamp(2rem,4vw,3.3rem)}.hero{padding:2rem}
@media(max-width:820px){aside{max-height:none}.layout{display:block}}
@media print{.section{break-inside:auto}.equation,pre,table,th,td{background:white;color:#202225;border-color:#ccc}.section>p>strong:first-child{color:#202225}.katex{color:#202225}.table-wrap{overflow:visible}table{min-width:0}.equation{break-inside:avoid}:not(pre)>code{background:#eee}th{color:#202225}}
</style></head><body><div class="layout">
<aside aria-label="Report navigation"><p class="nav-title">Comparison</p><nav>${navigation.join('')}</nav></aside>
<main><header class="hero"><p class="eyebrow">17 September 2026 · Math and code</p>
<h1>QuantLet IRMF and imf</h1><p class="lede">The same robust filtering family, with different kernels, reference targets and numerical solvers.</p></header>
${sections.replace(/<table>/g, '<div class="table-wrap"><table>').replace(/<\/table>/g, '</table></div>')}
<footer>${displayCount} display equations and ${inlineCount} inline expressions rendered with KaTeX ${katex.version}. Math, fonts and figures are embedded; no network connection or JavaScript is required to read the report.</footer>
</main></div></body></html>`;
if (/\{\{[^}]+\}\}/.test(html)) throw new Error('Unresolved template placeholder');
fs.mkdirSync(reportDir, { recursive: true });
fs.writeFileSync(path.join(reportDir, 'quantlet-imf-comparison.html'), html);
const reports = fs.readdirSync(reportDir).filter(name => !name.startsWith('.') && name.endsWith('.html')).sort();
const cards = reports.map(name => {
  const page = fs.readFileSync(path.join(reportDir, name), 'utf8');
  const title = page.match(/<title>(.*?)<\/title>/s)?.[1] || escape(name);
  return '<a class="report" href="reports/' + encodeURIComponent(name) + '"><strong>' + title + '</strong><span>' + escape(name) + '</span></a>';
});
fs.writeFileSync(path.join(repo, 'tmp/index.html'), `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>imf reports</title><style>body{color:#edf0f2;background:#202225;font:17px/1.6 system-ui;margin:3rem auto;padding:0 1.5rem;max-width:60rem}.report{display:block;background:#292c31;border:1px solid #454a52;border-radius:12px;padding:1.3rem;margin:1rem 0;color:#8ab4f8;text-decoration:none}.report span{display:block;color:#b2b8c0;font-size:.85rem}a:focus-visible{outline:2px solid #8ab4f8;outline-offset:4px}</style></head><body><h1>imf reports</h1>${cards.join('')}</body></html>`);
console.log(JSON.stringify({ report: path.join(reportDir, 'quantlet-imf-comparison.html'), displayCount, inlineCount, bytes: Buffer.byteLength(html) }));
