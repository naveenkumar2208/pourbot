const $ = id => document.getElementById(id);
const post = (url, body) => fetch(url, {
  method: "POST",
  headers: {"x-pourbot": "1", "content-type": "application/json"},
  body: body ? JSON.stringify(body) : undefined,
}).then(async r => { if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); });

let state = null, checkItems = {}, socketOpen = false, builtCameras = false, overlaysOn = {};
const fmt = (v, d = 0) => v == null ? "-" : Number(v).toFixed(d);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

fetch("/api/checks").then(r => r.json()).then(c => { checkItems = c; });

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { socketOpen = true; };
  ws.onmessage = e => { state = JSON.parse(e.data); render(); };
  ws.onclose = () => {
    socketOpen = false;
    $("connection").className = "pill bad";
    $("connection").textContent = "App not responding";
    $("run").disabled = true;
    setTimeout(connect, 1000);
  };
}
connect();

const act = fn => async (...a) => { try { await fn(...a); } catch (e) { alert(e.message); } };
const stopNow = () => post("/api/stop").catch(e => alert("STOP FAILED: " + e.message + ". Use the hardware e-stop."));
$("stop").onclick = stopNow;
document.addEventListener("keydown", e => {
  if (e.code === "Space" && !["INPUT", "BUTTON"].includes(e.target.tagName)) { e.preventDefault(); stopNow(); }
});
$("clear").onclick = act(() => post("/api/clear-stop"));
$("home").onclick = act(async () => {
  const p = await post("/api/home/plan");
  const verb = state.dry_run ? "Simulate moving" : "Move";
  const lost = state.current > 2 ? "\n\nThis abandons the current run; you will start again from measuring." : "";
  if (confirm(`${verb} the arm to home?\n\nAbout ${p.seconds} s, largest joint change ${p.biggest_joint_change}\u00b0.\n${p.note}.${lost}`)) await post("/api/home/run");
});
$("restart").onclick = act(async () => {
  if (confirm("Restart from the first step? Nothing moves, but all measurements and plans are discarded.")) await post("/api/restart");
});
$("run").onclick = act(() => post(`/api/run/${state.steps[state.current].id}`));

function render() {
  const s = state, st = s.status;
  $("mode").hidden = !s.dry_run;
  $("connection").className = "pill " + (st.connected ? "good" : "bad");
  $("connection").textContent = st.connected ? "Robot connected" : "Robot offline";
  $("motion").className = "pill " + (st.moving ? "busy" : "");
  $("motion").textContent = st.moving ? "Arm moving" : "Arm still";
  $("grip").textContent = st.gripper_position == null ? "Gripper -" : `Gripper ${(st.gripper_position / 10).toFixed(0)} mm` + (st.holding ? ", holding" : "");
  $("stopped").hidden = !s.stopped;

  $("steps").innerHTML = s.steps.map((x, i) =>
    `<li class="${x.state}" ${i === s.current ? 'aria-current="step"' : ""}><span class="n">${x.state === "done" ? "✓" : i + 1}</span>${esc(x.title)}</li>`).join("");

  const finished = s.current >= s.steps.length;
  const step = finished ? null : s.steps[s.current];
  $("stepcount").textContent = finished ? "Finished" : `Step ${s.current + 1} of ${s.steps.length}`;
  $("title").textContent = finished ? "Pour complete" : step.title;
  $("text").textContent = finished ? "The flask is back and the arm is at home. Check the result, then press Restart for another run." : step.text;

  renderChecklist(step);
  renderPlan(step);
  renderProgress();

  const previous = s.current > 0 ? s.steps[s.current - 1] : null;
  const result = previous && previous.result;
  $("result").hidden = !result;
  if (result) $("result").textContent = `${previous.title}: ${result}`;

  const failureLine = s.failed ? [...s.log].reverse().find(l => l.text.startsWith("FAILED")) : null;
  $("failure").hidden = !failureLine;
  if (failureLine) $("failure").textContent = failureLine.text.replace("FAILED: ", "") + " Fix the cause, then press Run to try this step again.";

  const needsChecks = step && step.kind === "check" && !Object.keys(checkItems).every(k => s.checks[k]);
  const needsPlan = step && step.kind === "motion" && !step.plan && !["startpos", "endpos"].includes(step.id);
  $("run").disabled = finished || !!s.busy || s.stopped || !st.connected || !socketOpen || needsChecks || needsPlan;
  $("run").textContent = !step ? "Done" : s.busy ? "Running…" :
    step.kind === "motion" ? (s.dry_run ? "Simulate this move" : "Move the arm") :
    step.kind === "gripper" ? (s.dry_run ? "Simulate gripper" : "Move the gripper") :
    step.kind === "check" ? "Confirm" : step.kind === "plan" ? "Rehearse now" :
    step.kind === "hold" ? "Start drain timer" : "Measure now";
  $("restart").disabled = !!s.busy;
  $("home").disabled = !!s.home_blocked;
  $("home").title = s.home_blocked || "Move the arm to the saved home position";

  renderCameras();
  renderScene();
  renderMaps(step);
  renderLog();
}

function renderChecklist(step) {
  const box = $("checklist");
  if (!step || step.kind !== "check") { box.innerHTML = ""; delete box.dataset.built; return; }
  if (!box.dataset.built && Object.keys(checkItems).length) {
    box.innerHTML = Object.entries(checkItems).map(([k, v]) => `<label><input type="checkbox" data-key="${k}"><span>${esc(v)}</span></label>`).join("");
    box.dataset.built = "1";
    box.querySelectorAll("input").forEach(i => i.onchange = act(() => post(`/api/check/${i.dataset.key}/${i.checked ? 1 : 0}`)));
  }
  box.querySelectorAll("input").forEach(i => { i.checked = !!state.checks[i.dataset.key]; });
}

function renderPlan(step) {
  const p = step && step.plan;
  if (!p) { $("plan").innerHTML = ""; return; }
  const t = p.targets.at(-1);
  const facts = [["Time", `${p.seconds} s`], ["Largest joint change", `${p.biggest_joint_change}°`], ["Waypoints", p.waypoints]];
  if (t) facts.push(["Ends at (mm)", `${fmt(t[0])}, ${fmt(t[1])}, ${fmt(t[2])}`]);
  if (p.note) facts.unshift(["Plan", p.note]);
  $("plan").innerHTML = facts.map(([a, b]) => `<div><dt>${a}</dt><dd>${esc(b)}</dd></div>`).join("");
}

function renderProgress() {
  const s = state, bar = $("progress");
  let share = null;
  if (s.progress && s.progress.total) share = s.progress.done / s.progress.total;
  if (s.hold_left != null) share = 1 - s.hold_left / s.drain;
  bar.hidden = share == null && !s.busy;
  bar.firstElementChild.style.width = share == null ? "100%" : `${Math.min(100, share * 100)}%`;
  bar.firstElementChild.style.opacity = share == null ? ".35" : "1";
}

function renderCameras() {
  const box = $("cameras");
  if (!builtCameras) {
    box.innerHTML = state.cameras.map(name => `<div class="cam">
      <div class="camhead"><h2>Camera: ${esc(name)} <span class="feed" data-feed="${esc(name)}"></span></h2>
      <label><input type="checkbox" data-overlay="${esc(name)}"> Show last measurement</label></div>
      <img id="cam-${esc(name)}" alt="Live view from camera ${esc(name)}"></div>`).join("");
    box.querySelectorAll("input").forEach(i => { i.onchange = () => { overlaysOn[i.dataset.overlay] = i.checked; }; });
    builtCameras = true;
    state.cameras.forEach(startFeed);
  }
  // Show a measurement right after it is taken; any step that follows goes back to the live feed.
  if (state.fresh_measurement !== renderCameras.fresh) {
    renderCameras.fresh = state.fresh_measurement;
    state.cameras.forEach(n => { overlaysOn[n] = state.fresh_measurement && state.overlays.includes(n); });
  }
  state.cameras.forEach(n => {
    const frozen = overlaysOn[n] && state.overlays.includes(n);
    box.querySelector(`[data-overlay="${n}"]`).checked = !!overlaysOn[n];
    const label = box.querySelector(`[data-feed="${n}"]`);
    label.textContent = frozen ? "Last measurement, not live" : "Live";
    label.className = "feed " + (frozen ? "frozen" : "live");
  });
}

function startFeed(name) {
  const img = $(`cam-${name}`);
  const next = () => {
    const showOverlay = overlaysOn[name] && state.overlays.includes(name);
    img.src = `/api/${showOverlay ? "overlay" : "camera"}/${name}?t=${Date.now()}`;
  };
  img.onload = () => setTimeout(next, overlaysOn[name] ? 500 : 400);
  img.onerror = () => setTimeout(next, 2000);
  next();
}

function renderScene() {
  const sc = state.scene;
  if (!sc || !sc.flask || !sc.cup) { $("measures").textContent = sc && sc.problems.length ? sc.problems.join("; ") : "Not measured yet."; $("nudges").innerHTML = ""; return; }
  const row = (n, o, z) => `<tr><td>${n}</td><td>${fmt(o.x)}</td><td>${fmt(o.y)}</td><td>${fmt(o[z])}</td><td>${fmt(o.radius * 2)}</td></tr>`;
  const warns = [];
  if (sc.table_tilt_deg > 1) warns.push(`The table looks ${fmt(sc.table_tilt_deg, 1)}° tilted to the camera. It may be camera calibration; watch the first grip closely.`);
  if (sc.flask.height < 90) warns.push("Flask height looks low.");
  for (const [name, mm] of Object.entries(sc.disagreement_mm)) if (mm > 8) warns.push(`Cameras disagree on the ${name} by ${fmt(mm)} mm.`);
  $("measures").innerHTML = `<table><tr><th>mm</th><th>x</th><th>y</th><th>z</th><th>width</th></tr>
    ${row("Flask mouth", sc.flask, "rim_z")}${row("Cup rim", sc.cup, "rim_z")}
    <tr><td>Table</td><td></td><td></td><td>${fmt(sc.table_z)}</td><td></td></tr></table>
    ${sc.obstacles.length ? `<div class="warn">${sc.obstacles.length} other object${sc.obstacles.length > 1 ? "s" : ""} on the table will be avoided.</div>` : ""}
    ${warns.map(w => `<div class="warn">${esc(w)}</div>`).join("")}`;

  const idx = state.steps.findIndex(s => s.id === "approach");
  const open = state.current <= idx && !state.busy;
  const box = $("nudges");
  const key = open ? "open" : "closed";
  if (box.dataset.key !== key) {
    box.dataset.key = key;
    box.innerHTML = !open ? "" : ["flask", "cup"].map(item => `<div class="nudge"><b>${item === "flask" ? "Flask" : "Cup"}</b>
      ${["x", "y", "z"].map(a => `<span>${a}
        <button data-i="${item}" data-a="${a}" data-m="-2" aria-label="${item} ${a} minus 2 millimetres">−2</button>
        <button data-i="${item}" data-a="${a}" data-m="2" aria-label="${item} ${a} plus 2 millimetres">+2</button></span>`).join("")}
    </div>`).join("");
    box.querySelectorAll("button").forEach(b => { b.onclick = act(() => post("/api/nudge", {item: b.dataset.i, axis: b.dataset.a, mm: +b.dataset.m})); });
  }
}

function renderMaps(step) {
  const sc = state.scene, st = state.status;
  const topX = x => (x - 150) * 400 / 800, topY = y => 150 - y * 300 / 600;
  const sideX = topX, sideZ = z => 270 - (z + 20) * 0.5;
  let top = `<line x1="${topX(0)}" y1="0" x2="${topX(0)}" y2="300" stroke="#e3e6e1"/>`;
  let side = `<line x1="0" y1="${sideZ(0)}" x2="400" y2="${sideZ(0)}" stroke="#b9bfb8"/>`;
  top += `<text x="6" y="294">x increases to the right; base is off the left edge</text>`;
  if (sc && sc.flask && sc.cup && state.layout) {
    const l = state.layout, f = sc.flask, c = sc.cup;
    top += `<circle cx="${topX(f.x)}" cy="${topY(f.y)}" r="${l.base_r / 2}" fill="#1f6fd122" stroke="var(--flask)" stroke-width="2"/>`;
    top += `<circle cx="${topX(c.x)}" cy="${topY(c.y)}" r="${l.cup_half / 2}" fill="#c8322b22" stroke="var(--cup)" stroke-width="2"/>`;
    for (const b of sc.obstacles) top += `<rect x="${topX(b.x0)}" y="${topY(b.y1)}" width="${(b.x1 - b.x0) / 2}" height="${(b.y1 - b.y0) / 2}" fill="#8882" stroke="#888"/>`;
    side += `<rect x="${sideX(f.x) - l.base_r / 2}" y="${sideZ(f.rim_z)}" width="${l.base_r}" height="${sideZ(sc.table_z) - sideZ(f.rim_z)}" fill="#1f6fd122" stroke="var(--flask)" stroke-width="2"/>`;
    side += `<rect x="${sideX(c.x) - l.cup_half / 2}" y="${sideZ(c.rim_z)}" width="${l.cup_half}" height="${sideZ(sc.table_z) - sideZ(c.rim_z)}" fill="#c8322b22" stroke="var(--cup)" stroke-width="2"/>`;
  }
  const targets = (step && step.plan && step.plan.targets) || [];
  targets.forEach(t => {
    top += `<circle cx="${topX(t[0])}" cy="${topY(t[1])}" r="4" fill="var(--target)"/>`;
    side += `<circle cx="${sideX(t[0])}" cy="${sideZ(t[2])}" r="4" fill="var(--target)"/>`;
  });
  const g = st.gripper_pose;
  if (g) {
    top += `<circle cx="${topX(g.x)}" cy="${topY(g.y)}" r="5" fill="var(--grip)"/>`;
    side += `<circle cx="${sideX(g.x)}" cy="${sideZ(g.z)}" r="5" fill="var(--grip)"/>`;
  }
  $("top").innerHTML = top;
  $("side").innerHTML = side;
}

function renderLog() {
  const box = $("log"), atEnd = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
  const key = state.log.length + (state.log.at(-1)?.t || "");
  if (renderLog.key === key) return;
  renderLog.key = key;
  box.innerHTML = state.log.map(l => `<div class="${l.text.startsWith("FAILED") || l.text.startsWith("STOP") ? "fail" : ""}"><time>${l.t}</time>${esc(l.text)}</div>`).join("");
  if (atEnd) box.scrollTop = box.scrollHeight;
}
