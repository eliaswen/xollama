#!/bin/bash
# G4, decision models: /v1/systemone on a side server on the engine this tree
# pins. Since v0.40.4 every GGUF decision model is one POST /v1/systemone to its
# engine (llama.cpp b11521 native). nimble and a Clef model (clef-flash, which
# also lists `vision`), both in the format v0.40.4 pulls, must be served: by
# opencoti when the pin declares systemone_v1, else by stock llama-server, each
# routing logged (server/decision_engine.go). WANT=opencoti|stock (default
# stock: no pinned engine declares systemone_v1 yet).
#   STORE=<scratch store with nimble and clef-flash, pulled by v0.40.4> [X=<xollama in an assembled rootfs>] [W=] [WANT=] scripts/gates/decision.sh
. "$(dirname "$0")/side.sh"
D=$here/decision
up "${STORE:?a writable scratch store}" serve-decision.log
ask() { # ask <model> <request file> <text the answer must hold>
  out=$(sed "s/\"model\":\"[A-Za-z-]*\"/\"model\":\"$1\"/" "$D/$2" | curl -s -m 900 -w ' http=%{http_code}' http://$H/v1/systemone -H 'Content-Type: application/json' -d @-)
  case "$out" in *"$3"*"http=200") echo "PASS $1 $2: $(echo "$out" | cut -c1-200)";; *) echo "FAIL $1 $2: $(echo "$out" | cut -c1-300)";; esac
}
for m in nimble clef-flash; do
  curl -s http://$H/api/tags | python3 -c "import json,sys; c=[x.get('capabilities') for x in json.load(sys.stdin)['models'] if x['name']=='$m:latest']; print(('PASS' if c and all('decision' in k for k in c) else 'FAIL'),'$m listed with',c)"
  ask $m req.json '"choice":"bug"'
  ask $m req-multi.json '"type":"noul"'
done
ask clef-flash req-image.json '"choice":"red"'
stock=$(grep -a 'starting llama-server' serve-decision.log | grep -c 'lib/ollama/llama-server')
routed=$(grep -a -c "decision model is served by stock llama-server" serve-decision.log)
echo "engines: opencoti loads $(grep -a -c 'opencoti build' serve-decision.log), stock loads $stock, routed to stock (logged) $routed"
if [ "${WANT:-stock}" = stock ]; then
  [ "$stock" -gt 0 ] && [ "$routed" -eq 2 ] && echo "PASS served by stock, both routings logged" || echo "FAIL want stock loads and 2 routing lines"
else
  [ "$stock" -eq 0 ] && [ "$routed" -eq 0 ] && echo "PASS served by opencoti" || echo "FAIL want no stock load"
fi
down; echo DECISION-DONE
