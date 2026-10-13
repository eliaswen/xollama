#!/bin/bash
# N runs of 512 tokens against a running xollama; prints tok/s per run, then
# one greedy run (temperature 0). A FAIL line is a run that: returned an error;
# answered no tokens; or answered fewer than 20 different words. bug-4048
# (Qwen3.5-family on Metal with c11) was all three: sampled, one token repeated
# to the limit at full speed; greedy, an engine exception that reached the
# client as an empty 200 until the `engine-stream-error` hook.
H=${H:-127.0.0.1:22434}
P="Write a detailed essay about the history of the printing press, its inventors, and its effects on European society."
judge() {
  python3 -c "
import sys,json
raw=sys.stdin.read()
try: d=json.loads(raw)
except Exception: print('$1', 'FAIL: not a JSON answer: %r'%raw[:80]); sys.exit()
e=d.get('error'); t=(d.get('response') or '')+(d.get('thinking') or ''); n=d.get('eval_count')
if e: print('$1', 'FAIL: error:', e)
elif not n: print('$1', 'FAIL: no tokens and no error')
elif len(set(t.split()))<20: print('$1', 'FAIL: %d tokens but %d different words: %r'%(n, len(set(t.split())), t[:60]))
else: print('$1', 'tokens %d  %.1f tok/s  prompt %.0f tok/s  load %.1fs'%(n, n/d['eval_duration']*1e9, d['prompt_eval_count']/max(d['prompt_eval_duration'],1)*1e9, d['load_duration']/1e9))"
}
for m in "$@"; do
  for i in 1 2 3 4; do
    curl -s http://$H/api/generate -d "{\"model\":\"$m\",\"prompt\":\"$P\",\"stream\":false,\"options\":{\"num_predict\":512,\"seed\":$i,\"temperature\":0.7}}" | judge "$m run $i"
  done
  curl -s http://$H/api/generate -d "{\"model\":\"$m\",\"prompt\":\"$P\",\"stream\":false,\"options\":{\"num_predict\":512,\"seed\":1,\"temperature\":0}}" | judge "$m greedy"
  curl -s http://$H/api/ps | python3 -c "import sys,json
for m in json.load(sys.stdin)['models']: print('  ps', m['name'], 'vram %.1f GB of %.1f GB'%(m['size_vram']/1e9, m['size']/1e9), 'ctx', m.get('context_length'))"
done
