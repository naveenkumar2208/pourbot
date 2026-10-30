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
  // Space must stop the arm even when a button still has focus from being clicked.
  if (e.code === "Space" && e.target.tagName !== "INPUT") { e.preventDefault(); stopNow(); }
});
$("clear").onclick = act(() => post("/api/clear-stop"));
$("home").onclick = act(async () => {
  const p = await post("/api/home/plan");
  const verb = state.dry_run ? "Simulate moving" : "Move";
  const lost = state.current > 2 ? "\n\nThis abandons the current run; you will start again from measuring." : "";
  if (confirm(`${verb} the arm to home?\n\nAbout ${p.seconds} s, largest joint change ${p.biggest_joint_change}\u00b0.\n${p.note}.${lost}`)) await post("/api/home/run");
});
$("gripopen").onclick = act(async () => {
  if (state.status.holding || state.fingers_around) {
    if (!confirm("The fingers may be holding a container. Opening could drop it. Open anyway?")) return;
  }
  await post("/api/gripper/open");
});
$("gripclose").onclick = act(async () => {
  if (confirm("Close the gripper now? Keep your hands clear of the fingers.")) await post("/api/gripper/close");
});
$("back").onclick = act(async () => {
  const prev = state.steps[state.current - 1];
  const moves = prev.kind === "motion" || prev.kind === "gripper";
  const detail = !moves ? "Nothing moves." : prev.kind === "motion" ? "The arm retraces that move in reverse." : prev.id.endsWith("grab") ? "The gripper opens." : prev.id.endsWith("release") ? "The gripper closes." : "Nothing moves.";
  if (confirm(`Go back to "${prev.title}"?\n\n${detail}`)) await post("/api/back");
});
$("restart").onclick = act(async () => {
  if (confirm("Restart from the first step? Nothing moves, but all measurements and plans are discarded.")) await post("/api/restart");
});
$("run").onclick = act(() => post(`/api/run/${state.steps[state.current].id}`));
$("play").onclick = act(async () => {
  if (state.playing) { await post("/api/pause"); return; }
  const action = state.dry_run ? "simulate" : "move the arm through";
  if (confirm(`Brew will ${action} every remaining step without asking again. It pauses once for you to choose your potions.\n\nKeep a hand on the hardware e-stop. It ends on its own if a step fails, and Space or STOP halts it.`)) await post("/api/play");
});

function render() {
  const s = state, st = s.status;
  $("mode").hidden = !s.dry_run;
  $("connection").className = "pill " + (st.connected ? "good" : "bad");
  $("connection").textContent = st.connected ? "Robot connected" : "Robot offline";
  $("motion").className = "pill " + (st.moving ? "busy" : "");
  $("motion").textContent = st.moving ? "Arm moving" : "Arm still";
  $("grip").textContent = st.gripper_position == null ? "Gripper -" : `Gripper ${(st.gripper_position / 10).toFixed(0)} mm` + (st.holding ? ", holding" : "");
  $("stopped").hidden = !s.stopped;
  const rec = s.recording;
  $("recording").hidden = !rec || !rec.enabled;
  if (rec && rec.enabled) {
    $("recording").className = "pill " + (rec.error ? "bad" : rec.active ? "busy" : "");
    $("recording").textContent = rec.error ? `Not saving: ${rec.error}` :
      rec.active ? `Recording ${rec.samples}${rec.dropped ? `, ${rec.dropped} lost` : ""}` : "Recorder ready";
  }

  $("steps").innerHTML = s.steps.map((x, i) =>
    `<li class="${x.state}" ${i === s.current ? 'aria-current="step"' : ""}><span class="n">${x.state === "done" ? "✓" : i + 1}</span>${esc(x.title)}</li>`).join("");

  const finished = s.current >= s.steps.length;
  const step = finished ? null : s.steps[s.current];
  $("stepcount").textContent = finished ? "Finished" : `Step ${s.current + 1} of ${s.steps.length}`;
  $("title").textContent = finished ? "The potion is brewed" : step.title;
  $("text").textContent = finished ? "The containers are back and the arm is at home. Check the result, then press Restart for another brew." : step.text;

  renderChecklist(step);
  renderChooser(step);
  renderCauldron(finished);
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
  $("run").disabled = finished || !!s.busy || s.playing || s.stopped || !st.connected || !socketOpen || needsChecks || needsPlan || (step && step.kind === "choose" && !s.selection.length);
  const needsChoice = step && step.kind === "choose" && !s.selection.length;
  $("play").disabled = !s.playing && (finished || !!s.busy || s.stopped || !st.connected || !socketOpen || needsChecks || needsChoice);
  $("play").innerHTML = s.playing ? "&#10074;&#10074; Pause after this step" : "&#9654; Brew potion";
  $("play").className = "play" + (s.playing ? " on" : "");
  $("run").textContent = !step ? "Done" : s.busy ? "Running…" :
    step.kind === "motion" ? (s.dry_run ? "Simulate this move" : "Move the arm") :
    step.kind === "gripper" ? (s.dry_run ? "Simulate gripper" : "Move the gripper") :
    step.kind === "check" ? "Confirm" : step.kind === "plan" ? "Rehearse now" :
    step.kind === "choose" ? "Use these potions" : "Measure now";
  $("restart").disabled = !!s.busy || s.playing;
  for (const id of ["gripopen", "gripclose"]) {
    $(id).disabled = !!s.manual_blocked;
    $(id).title = s.manual_blocked || "";
  }
  $("back").disabled = !!s.back_blocked;
  $("back").title = s.back_blocked || "Undo the last finished step";
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

function renderChooser(step) {
  const box = $("chooser"), sc = state.scene;
  if (!step || step.kind !== "choose" || !sc || !sc.vessels) { box.innerHTML = ""; delete box.dataset.key; return; }
  const key = JSON.stringify([sc.vessels.map(v => [v.id, v.profile, Math.round(v.height)]), state.selection, !!state.busy || state.playing]);
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  box.innerHTML = `<div class="potions">${sc.vessels.map(v => {
    const at = state.selection.indexOf(v.id);
    const note = v.usable ? `${esc(v.profile)}, ${fmt(v.height)} mm tall` : `${fmt(v.height)} mm tall, needs setup in settings.json`;
    return `<button class="potion ${at >= 0 ? "on" : ""}" data-id="${esc(v.id)}" ${v.usable ? "" : "disabled"} aria-pressed="${at >= 0}">
      <span class="num">${at >= 0 ? at + 1 : ""}</span><span><b>Container ${esc(v.id)}</b><small>${note}</small></span></button>`;
  }).join("")}</div>`;
  box.querySelectorAll("button").forEach(b => {
    b.onclick = act(() => {
      const id = b.dataset.id, ids = state.selection.includes(id) ? state.selection.filter(x => x !== id) : [...state.selection, id];
      return post("/api/select", {ids});
    });
  });
}

const NAMES = [["Moonlit", "Elixir"], ["Cursed", "Draught"], ["Howling", "Tonic"], ["Bewitched", "Brew"], ["Midnight", "Philter"], ["Phantom", "Nectar"], ["Grim", "Cordial"], ["Hexed", "Serum"]];

function renderCauldron(finished) {
  const s = state, total = s.selection.length || 1, share = s.poured / total;
  $("liquid").setAttribute("y", 148 - 74 * share);
  $("cauldron").classList.toggle("brewing", s.pouring || s.playing);
  const seed = [...s.selection.join("")].reduce((n, c) => n * 7 + c.charCodeAt(0), 3);
  const name = NAMES[seed % NAMES.length].join(" ");
  $("brewline").textContent = finished && s.selection.length ? `You brewed: ${name}` :
    !s.selection.length ? "Choose your potions" : s.pouring ? "The cauldron bubbles..." : `${s.poured} of ${s.selection.length} poured`;
  $("recipe").innerHTML = s.selection.map((id, i) =>
    `<li class="${i < s.poured ? "done" : ""}">${i < s.poured ? "\u2713" : `${i + 1}.`} Container ${esc(id)}</li>`).join("");
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
  if (!sc || !sc.vessels || !sc.vessels.length || !sc.cup) { $("measures").textContent = sc && sc.problems.length ? [...new Set(sc.problems)].join("; ") : "Not measured yet."; $("nudges").innerHTML = ""; delete $("nudges").dataset.key; return; }
  const row = (n, o) => `<tr><td>${n}</td><td>${fmt(o.x)}</td><td>${fmt(o.y)}</td><td>${fmt(o.rim_z)}</td><td>${fmt(o.radius * 2)}</td></tr>`;
  const warns = [];
  if (sc.table_tilt_deg > 1) warns.push(`The table looks ${fmt(sc.table_tilt_deg, 1)}\u00b0 tilted to the camera. It may be camera calibration; watch the first grip closely.`);
  for (const [name, mm] of Object.entries(sc.disagreement_mm)) if (mm > 8) warns.push(`Cameras disagree on the ${name} by ${fmt(mm)} mm.`);
  $("measures").innerHTML = `<table><tr><th>mm</th><th>x</th><th>y</th><th>z</th><th>width</th></tr>
    ${sc.vessels.map(v => row(`Container ${esc(v.id)}`, v)).join("")}${row("Cup rim", sc.cup)}
    <tr><td>Table</td><td></td><td></td><td>${fmt(sc.table_z)}</td><td></td></tr></table>
    ${sc.obstacles.length ? `<div class="warn">${sc.obstacles.length} other object${sc.obstacles.length > 1 ? "s" : ""} on the table will be avoided.</div>` : ""}
    ${warns.map(w => `<div class="warn">${esc(w)}</div>`).join("")}`;

  const box = $("nudges");
  const open = state.can_change_positions && !state.busy && !state.playing;
  const items = [...sc.vessels.map(v => [v.id, `Container ${v.id}`]), ["cup", "Cup"]];
  const key = open ? items.map(i => i[0]).join() : "closed";
  if (box.dataset.key !== key) {
    box.dataset.key = key;
    box.innerHTML = !open ? "" : items.map(([item, label]) => `<div class="nudge"><b>${esc(label)}</b>
      ${["x", "y", "z"].map(a => `<span>${a}
        <button data-i="${esc(item)}" data-a="${a}" data-m="-2" aria-label="${esc(label)} ${a} minus 2 millimetres">\u22122</button>
        <button data-i="${esc(item)}" data-a="${a}" data-m="2" aria-label="${esc(label)} ${a} plus 2 millimetres">+2</button></span>`).join("")}
    </div>`).join("");
    box.querySelectorAll("button").forEach(b => { b.onclick = act(() => post("/api/nudge", {item: b.dataset.i, axis: b.dataset.a, mm: +b.dataset.m})); });
  }
}

function renderMaps(step) {
  const sc = state.scene, st = state.status;
  const topX = x => (x - 150) * 400 / 800, topY = y => 150 - y * 300 / 600;
  const sideX = topX, sideZ = z => 270 - (z + 20) * 0.5;
  let top = `<line x1="${topX(0)}" y1="0" x2="${topX(0)}" y2="300" stroke="#3b2a5c"/>`;
  let side = `<line x1="0" y1="${sideZ(0)}" x2="400" y2="${sideZ(0)}" stroke="#6d5a90"/>`;
  top += `<text class="hint" x="6" y="294">x increases to the right; base is off the left edge</text>`;
  if (sc && sc.vessels && sc.cup) {
    const c = sc.cup, half = state.cup_half;
    for (const v of sc.vessels) {
      const r = (v.base_diameter || 100) / 2, picked = state.selection.includes(v.id);
      const color = picked ? "var(--picked)" : "var(--flask)";
      top += `<circle cx="${topX(v.x)}" cy="${topY(v.y)}" r="${r / 2}" fill="none" stroke="${color}" stroke-width="2"/><text x="${topX(v.x) - 4}" y="${topY(v.y) + 4}">${esc(v.id)}</text>`;
      side += `<rect x="${sideX(v.x) - r / 2}" y="${sideZ(v.rim_z)}" width="${r}" height="${sideZ(sc.table_z) - sideZ(v.rim_z)}" fill="none" stroke="${color}" stroke-width="2"/>`;
    }
    top += `<circle cx="${topX(c.x)}" cy="${topY(c.y)}" r="${half / 2}" fill="#ff5a5033" stroke="var(--cup)" stroke-width="2"/>`;
    for (const b of sc.obstacles) top += `<rect x="${topX(b.x0)}" y="${topY(b.y1)}" width="${(b.x1 - b.x0) / 2}" height="${(b.y1 - b.y0) / 2}" fill="#8882" stroke="#888"/>`;
    side += `<rect x="${sideX(c.x) - half / 2}" y="${sideZ(c.rim_z)}" width="${half}" height="${sideZ(sc.table_z) - sideZ(c.rim_z)}" fill="#ff5a5033" stroke="var(--cup)" stroke-width="2"/>`;
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
