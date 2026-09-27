"use strict";

const RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "A"];
const DECK_ORDER = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T"];
const KEY_HINT = { T: "0", A: "1" };
const KEY_MAP = { a: "A", 1: "A", 0: "T", t: "T", j: "T", q: "T", k: "T" };
for (let i = 2; i <= 9; i++) KEY_MAP[String(i)] = String(i);
const ZONES = ["player", "dealer", "table"];
const ACTION_LABELS = {
  Stand: ["Plantarse", "Stand"],
  Hit: ["Pedir", "Hit"],
  Double: ["Doblar", "Double"],
  Split: ["Separar", "Split"],
  Surrender: ["Rendirse", "Surrender"],
  Blackjack: ["Blackjack", "Plántate"],
  Bust: ["Pasado", "Bust"],
};
const RESULT_LABELS = { win: "Gana", push: "Empate", loss: "Pierde", blackjack: "BJ", surrender: "Rendida" };
const SAVE_ACTION_KEYS = { h: "Hit", s: "Stand", d: "Double", p: "Split", r: "Surrender" };
const SAVE_RESULT_KEYS = { w: "win", e: "push", l: "loss", b: "blackjack" };

const state = {
  zone: "player",
  prevZone: "player",
  cards: { player: [], dealer: [], table: [] },
  history: [], // {zone, card, replaced?}
  decision: null, // foto del momento de decidir (mano de 2 cartas + crupier)
  last: null, // última respuesta de /calculate
};
let requestSeq = 0;

const $ = (sel) => document.querySelector(sel);

// ---------------------------------------------------------------- Preferencias locales
const store = {
  get(key, fallback) {
    try {
      const v = localStorage.getItem(`bj.${key}`);
      return v === null ? fallback : JSON.parse(v);
    } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`bj.${key}`, JSON.stringify(value)); } catch { /* sin almacenamiento */ }
  },
};

// ---------------------------------------------------------------- API
async function api(path, body, method) {
  const res = await fetch(path, {
    method: method || (body === undefined ? "GET" : "POST"),
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `Error ${res.status}`);
  }
  return data;
}

function betSettings() {
  return {
    bankroll: Number($("#k-bankroll").value) || 0,
    kelly_fraction: Number($("#k-fraction").value) || 0.5,
    min_bet: Number($("#k-min").value) || 0,
    max_bet: Number($("#k-max").value) || null,
  };
}

function roundPayload() {
  return {
    player_cards: state.cards.player,
    dealer_upcard: state.cards.dealer[0] ?? null,
    dead_cards: state.cards.table,
  };
}

// ---------------------------------------------------------------- Entrada de cartas
function setZone(zone) {
  if (zone !== state.zone) state.prevZone = state.zone;
  state.zone = zone;
  document.querySelectorAll(".zone").forEach((el) => el.classList.toggle("active", el.dataset.zone === zone));
}

function addCard(card) {
  const zone = state.zone;
  const entry = { zone, card };
  if (zone === "dealer") {
    entry.replaced = state.cards.dealer[0] ?? null;
    state.cards.dealer = [card];
    setZone(state.prevZone === "dealer" ? "player" : state.prevZone);
  } else {
    state.cards[zone].push(card);
  }
  state.history.push(entry);
  flashKey(card);
  refresh({ revertOnError: true });
}

function undo() {
  const last = state.history.pop();
  if (!last) return;
  revertEntry(last);
  refresh();
}

function revertEntry(entry) {
  if (entry.zone === "dealer") {
    state.cards.dealer = entry.replaced ? [entry.replaced] : [];
  } else {
    const list = state.cards[entry.zone];
    const idx = list.lastIndexOf(entry.card);
    if (idx >= 0) list.splice(idx, 1);
  }
}

function removeChip(zone, index) {
  const [card] = state.cards[zone].splice(index, 1);
  for (let i = state.history.length - 1; i >= 0; i--) {
    const h = state.history[i];
    if (h.zone === zone && h.card === card) { state.history.splice(i, 1); break; }
  }
  refresh();
}

function resetRoundState() {
  state.cards = { player: [], dealer: [], table: [] };
  state.history = [];
  state.decision = null;
  setZone("player");
}

function clearRound() {
  resetRoundState();
  refresh();
}

function allRoundCards() {
  return [...state.cards.player, ...state.cards.dealer, ...state.cards.table];
}

async function newRound() {
  const all = allRoundCards();
  try {
    if (all.length) await api("/discard", { cards: all });
    clearRound();
  } catch (e) {
    toast(e.message);
  }
}

async function undoRound() {
  try {
    const data = await api("/undo-round", {});
    toast(`Devueltas al zapato: ${data.restored.join(" ") || "(ninguna)"}`, false);
    refresh();
  } catch (e) {
    toast(e.message);
  }
}

// ---------------------------------------------------------------- Cálculo
async function refresh({ revertOnError = false } = {}) {
  renderZones();
  const seq = ++requestSeq;
  const payload = roundPayload();
  try {
    const data = await api("/calculate", { ...payload, mode: $("#engine-mode").value, bet: betSettings() });
    if (seq !== requestSeq) return; // respuesta obsoleta
    state.last = data;
    // Foto de la decisión: se actualiza mientras la mano tiene 2 cartas y se congela al pedir.
    if (payload.player_cards.length === 2 && payload.dealer_upcard && data.best) {
      state.decision = {
        player_cards: [...payload.player_cards],
        dealer_upcard: payload.dealer_upcard,
        dead_cards: [...payload.dead_cards],
        best: data.best,
        bet: data.kelly ? data.kelly.recommended_bet : null,
      };
    } else if (payload.player_cards.length < 2) {
      state.decision = null;
    }
    renderResult(data);
  } catch (e) {
    if (seq !== requestSeq) return;
    toast(e.message);
    if (revertOnError && state.history.length) {
      revertEntry(state.history.pop());
      refresh();
    }
  }
}

// ---------------------------------------------------------------- Render
function chip(card, small = false) {
  const el = document.createElement("div");
  el.className = "chip" + (small ? " small" : "");
  el.textContent = card;
  return el;
}

function handTotal(cards) {
  let hard = 0, ace = false;
  for (const c of cards) {
    hard += c === "A" ? 1 : c === "T" ? 10 : Number(c);
    if (c === "A") ace = true;
  }
  if (ace && hard + 10 <= 21) return { total: hard + 10, soft: true };
  return { total: hard, soft: false };
}

function renderZones() {
  for (const zone of ZONES) {
    $(`#chips-${zone}`).replaceChildren(
      ...state.cards[zone].map((c, i) => {
        const el = chip(c, zone === "table");
        el.title = "Quitar";
        el.addEventListener("click", (ev) => { ev.stopPropagation(); removeChip(zone, i); });
        return el;
      })
    );
  }
  const p = state.cards.player;
  if (p.length) {
    const { total, soft } = handTotal(p);
    $("#player-total").textContent = (soft ? "S" : "") + total;
  } else {
    $("#player-total").textContent = "";
  }
  const n = state.cards.table.length;
  $("#table-count").textContent = n ? `${n}` : "";
}

const fmtPct = (x, digits = 2) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(digits)}%`;
const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)}%`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const money = (x) => (x == null ? "–" : Number(x).toLocaleString("es-ES", { maximumFractionDigits: 2 }));

function renderResult(data) {
  renderShoe(data.shoe, data.rules);
  renderEngine(data);
  $("#timing").textContent = data.elapsed_ms != null ? `${data.elapsed_ms} ms` : "";

  const rec = $("#recommendation");
  const list = $("#actions");
  const expl = $("#explanation");
  list.replaceChildren();
  if (!data.actions || !data.actions.length) {
    rec.className = "recommendation empty";
    rec.textContent = data.message || "Introduce tu mano y la carta del crupier";
    expl.classList.add("hidden");
  } else {
    const best = data.actions[0];
    const [es, en] = ACTION_LABELS[best.action] || [best.action, ""];
    rec.className = "recommendation" + (best.action === "Bust" ? " bust" : "");
    const extra = best.action === "Blackjack" ? "" : ` (${en})`;
    const hand = data.player.blackjack ? "blackjack" : `${data.player.soft ? "blando" : "duro"} ${data.player.total}`;
    rec.innerHTML = `${esc(es)}${esc(extra)}<small>EV ${fmtPct(best.ev)} · ${hand} vs ${esc(data.dealer.upcard)}</small>`;

    if (data.explanation) {
      expl.innerHTML = esc(data.explanation).replace(/⚠.*$/, (m) => `<span class="warn">${m}</span>`);
      expl.classList.remove("hidden");
    } else {
      expl.classList.add("hidden");
    }

    for (const a of data.actions) {
      const [aes, aen] = ACTION_LABELS[a.action] || [a.action, ""];
      const li = document.createElement("li");
      li.className = "action" + (a === best ? " best" : "");
      const w = Math.min(Math.abs(a.ev), 1) * 50;
      const color = a.ev >= 0 ? "var(--good)" : "var(--bad)";
      const left = a.ev >= 0 ? 50 : 50 - w;
      const wpl = a.win != null
        ? `W ${pct(a.win)} · P ${pct(a.push)} · L ${pct(a.loss)}
           <div class="wplbar"><i style="width:${a.win * 100}%"></i><i style="width:${a.push * 100}%"></i><i style="width:${a.loss * 100}%"></i></div>`
        : `<span title="Win% no disponible con el modelo ML">W/P/L –</span>`;
      li.innerHTML = `
        <div class="name">${esc(aes)}<small>${esc(aen)}</small></div>
        <div class="evbar"><i style="left:${left}%;width:${w}%;background:${color}"></i></div>
        <div class="wpl">${wpl}</div>
        <div class="ev ${a.ev >= 0 ? "pos" : "neg"}">${fmtPct(a.ev)}</div>`;
      list.appendChild(li);
    }
  }

  const ins = $("#insurance");
  if (data.insurance) {
    const i = data.insurance;
    ins.classList.remove("hidden");
    ins.innerHTML = `<b>Seguro:</b> P(10 oculta) = ${pct(i.ten_prob, 2)} · EV ${fmtPct(i.ev)} por unidad asegurada → ${i.recommended ? "<b style='color:var(--good)'>TOMAR</b>" : "no tomar"}`;
  } else {
    ins.classList.add("hidden");
  }

  const mlo = $("#ml-outcome");
  if (data.ml_outcome) {
    const o = data.ml_outcome;
    mlo.textContent = `Modelo (dataset Kaggle, estrategia del dataset): Win ${pct(o.Win ?? 0)} · Push ${pct(o.Push ?? 0)} · Loss ${pct(o.Loss ?? 0)}`;
    mlo.classList.remove("hidden");
  } else {
    mlo.classList.add("hidden");
  }

  renderDealer(data);
  renderKelly(data);
}

function renderEngine(data) {
  const badge = $("#engine-badge");
  const src = data.source;
  badge.textContent = src === "ml" ? "ML" : src === "exact" ? "Exacto" : data.model_available ? "modelo listo" : "sin modelo";
  badge.className = "badge" + (src ? ` ${src}` : "");
  badge.title = data.model_available ? "blackjack_model.pkl cargado" : "Entrena con: python train_model.py";
}

function dealerClass(key, playerTotal, playerBJ) {
  if (playerTotal == null) return "";
  if (key === "bust") return "green";
  if (key === "BJ") return playerBJ ? "yellow" : "red";
  const v = Number(key);
  if (v > playerTotal) return "red";
  if (v === playerTotal) return "yellow";
  return "green";
}

function renderDealer(data) {
  const bars = $("#dealer-probs");
  bars.replaceChildren();
  const d = data.dealer;
  if (!d || !d.probs) {
    bars.innerHTML = `<span class="muted">Sin carta del crupier.</span>`;
    $("#dealer-bj").textContent = "";
    return;
  }
  const p = data.player;
  const hasHand = p && p.cards.length >= 2 && !p.busted;
  const pt = hasHand ? p.total : null;
  const bj = d.blackjack_prob || 0;
  const enhc = data.rules && data.rules.enhc;

  // Con ENHC el blackjack del crupier sigue siendo posible: probabilidades incondicionales.
  const rows = Object.entries(d.probs).map(([k, v]) => [k, enhc ? v * (1 - bj) : v]);
  if (enhc && bj > 0) rows.push(["BJ", bj]);

  for (const [k, v] of rows) {
    const row = document.createElement("div");
    row.className = `bar ${dealerClass(k, pt, p && p.blackjack)}`;
    const label = k === "bust" ? "Bust" : k;
    row.innerHTML = `<span>${label}</span><div class="track"><div class="fill" style="width:${(v * 100).toFixed(2)}%"></div></div><span class="val">${pct(v, 2)}</span>`;
    bars.appendChild(row);
  }
  $("#dealer-bj").textContent = bj > 0
    ? (enhc ? `ENHC · P(BJ) ${pct(bj, 2)}` : `P(BJ) ${pct(bj, 2)} · condicionado a que el crupier no tenga BJ (peek)`)
    : (pt != null ? `tu mano: ${pt}` : "");
}

function renderKelly(data) {
  const adv = data.advantage;
  if (!adv) return;
  const a = adv.advantage;
  const advEl = $("#kelly-adv");
  advEl.textContent = fmtPct(a);
  advEl.style.color = a > 0 ? "var(--good)" : "var(--bad)";
  $("#kelly-adv-detail").textContent =
    `base ${fmtPct(adv.base_edge)} (${adv.base_source === "exact" ? "exacta" : "tabla"}) + composición ${fmtPct(adv.shift)}`;
  $("#st-adv").textContent = fmtPct(a);
  $("#st-adv").style.color = a > 0 ? "var(--good)" : "";

  const k = data.kelly;
  if (k) {
    $("#kelly-bet").textContent = money(k.recommended_bet);
    $("#kelly-bet-detail").textContent = k.at_minimum
      ? "sin ventaja → apuesta mínima"
      : `${pct(k.full_kelly_fraction * k.kelly_fraction, 2)} del bankroll (Kelly completo ${pct(k.full_kelly_fraction, 2)})`;
    $("#kelly-summary").textContent = `apostar ${money(k.recommended_bet)}`;
  } else {
    $("#kelly-bet").textContent = "–";
    $("#kelly-bet-detail").textContent = "introduce tu bankroll";
    $("#kelly-summary").textContent = `ventaja ${fmtPct(a)}`;
  }
}

function renderShoe(shoe, rules) {
  if (!shoe) return;
  $("#st-total").textContent = `${shoe.total}/${shoe.initial_total}`;
  $("#st-decks").textContent = shoe.decks_remaining.toFixed(2);
  $("#st-pen").textContent = `${(shoe.penetration * 100).toFixed(1)}%`;
  $("#st-rc").textContent = shoe.running_count > 0 ? `+${shoe.running_count}` : shoe.running_count;
  const tc = shoe.true_count;
  $("#st-tc").textContent = tc > 0 ? `+${tc.toFixed(2)}` : tc.toFixed(2);
  $("#st-tc").style.color = tc >= 1 ? "var(--good)" : tc <= -1 ? "var(--bad)" : "";

  if (rules) {
    const payout = { 1.5: "3:2", 1.2: "6:5", 1: "1:1" }[rules.blackjack_payout] || rules.blackjack_payout;
    $("#rules-line").textContent =
      `${rules.num_decks} barajas · ${rules.dealer_hits_soft17 ? "H17" : "S17"} · ${rules.double_after_split ? "DAS" : "no DAS"} · ` +
      `${rules.enhc ? "ENHC" : "peek"} · BJ ${payout}` + (rules.surrender ? " · LS" : "");
  }

  const deck = $("#deck");
  deck.replaceChildren();
  const frac = shoe.total / shoe.initial_total;
  for (const r of DECK_ORDER) {
    const left = shoe.remaining[r];
    const neutral = shoe.initial[r] * frac;
    const dev = neutral > 0 ? left / neutral - 1 : 0;
    const cell = document.createElement("div");
    cell.className = "deck-cell";
    const cls = dev > 0.005 ? "rich" : dev < -0.005 ? "poor" : "";
    cell.innerHTML = `<div class="rank">${r}</div><div class="cnt">${left}</div><div class="dev ${cls}">${fmtPct(dev, 1)}</div>`;
    deck.appendChild(cell);
  }
}

let toastTimer;
function toast(msg, isError = true) {
  const el = $("#toast");
  el.textContent = msg;
  el.style.background = isError ? "var(--bad)" : "var(--panel-2)";
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 2600);
}

function flashKey(card) {
  const btn = document.querySelector(`.keypad button[data-card="${card}"]`);
  if (!btn) return;
  btn.classList.add("pressed");
  setTimeout(() => btn.classList.remove("pressed"), 90);
}

// ---------------------------------------------------------------- Ventaja exacta
async function exactAdvantage() {
  const btn = $("#btn-exact-adv");
  btn.disabled = true;
  $("#exact-adv").textContent = "calculando…";
  try {
    const t0 = performance.now();
    const data = await api("/advantage/exact", roundPayload());
    const s = ((performance.now() - t0) / 1000).toFixed(1);
    $("#exact-adv").innerHTML = `Exacta: <b style="color:${data.advantage > 0 ? "var(--good)" : "var(--bad)"}">${fmtPct(data.advantage, 3)}</b> · estimada ${fmtPct(data.estimate.advantage, 3)} · ${s} s`;
  } catch (e) {
    $("#exact-adv").textContent = "";
    toast(e.message);
  } finally {
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------- Guardar mano
const saveState = { action: null, result: null, netEdited: false };

function openSave() {
  const snap = state.decision || (state.cards.player.length >= 2 && state.cards.dealer[0]
    ? { player_cards: [...state.cards.player], dealer_upcard: state.cards.dealer[0], dead_cards: [...state.cards.table], best: state.last?.best }
    : null);
  if (!snap) {
    toast("Introduce tu mano y la carta del crupier antes de guardar");
    return;
  }
  const best = snap.best === "Blackjack" ? "Stand" : snap.best;
  const [bestEs] = ACTION_LABELS[snap.best] || [snap.best];
  $("#save-summary").innerHTML =
    `Mano <b>${esc(snap.player_cards.join(" "))}</b> vs <b>${esc(snap.dealer_upcard)}</b>` +
    (state.cards.player.length > 2 ? ` → final ${esc(state.cards.player.join(" "))}` : "") +
    ` · recomendado: <b style="color:var(--good)">${esc(bestEs)}</b>`;
  document.querySelectorAll("#seg-action button").forEach((b) => b.classList.toggle("recommended", b.dataset.v === best));
  saveState.netEdited = false;
  selectSaveAction(best || "Stand");
  selectSaveResult(snap.best === "Blackjack" ? "blackjack" : null);
  $("#save-bet").value = snap.bet ?? state.last?.kelly?.recommended_bet ?? store.get("lastBet", "");
  $("#save-dialog").showModal();
}

function selectSaveAction(action) {
  saveState.action = action;
  document.querySelectorAll("#seg-action button").forEach((b) => b.classList.toggle("sel", b.dataset.v === action));
  if (action === "Surrender") selectSaveResult("surrender");
  else updateNet();
}

function selectSaveResult(result) {
  saveState.result = result;
  document.querySelectorAll("#seg-result button").forEach((b) => b.classList.toggle("sel", b.dataset.v === result));
  updateNet();
}

function updateNet() {
  if (saveState.netEdited) return;
  const payout = state.last?.rules?.blackjack_payout ?? 1.5;
  const stake = saveState.action === "Double" || saveState.action === "Split" ? 2 : 1;
  const net = { win: stake, loss: -stake, push: 0, blackjack: payout, surrender: -0.5 }[saveState.result];
  $("#save-net").value = net ?? "";
}

async function confirmSave() {
  if (!saveState.result) { toast("Elige el resultado (W / E / L / B)"); return; }
  const snap = state.decision || {
    player_cards: [...state.cards.player], dealer_upcard: state.cards.dealer[0], dead_cards: [...state.cards.table],
  };
  const bet = $("#save-bet").value === "" ? null : Number($("#save-bet").value);
  const net = $("#save-net").value === "" ? null : Number($("#save-net").value);
  try {
    await api("/history/save", {
      decision: { player_cards: snap.player_cards, dealer_upcard: snap.dealer_upcard, dead_cards: snap.dead_cards },
      round_cards: allRoundCards(),
      final_cards: state.cards.player,
      action_taken: saveState.action,
      result: saveState.result,
      bet,
      net_units: net,
      mode: $("#engine-mode").value,
    });
    if (bet != null) store.set("lastBet", bet);
    $("#save-dialog").close();
    resetRoundState();
    refresh();
    loadHistory();
    toast("Mano guardada · mesa limpia", false);
  } catch (e) {
    toast(e.message);
  }
}

// ---------------------------------------------------------------- Historial
async function loadHistory() {
  try {
    const { hands, stats } = await api("/history?limit=30");
    const body = $("#history-body");
    body.replaceChildren(
      ...hands.map((h) => {
        const tr = document.createElement("tr");
        const time = h.created_at ? new Date(h.created_at).toLocaleTimeString("es-ES", { hour: "2-digit", minute: "2-digit" }) : "";
        const dev = h.action_taken !== h.recommended_action;
        const net = h.net_units ?? 0;
        tr.innerHTML = `
          <td>${h.id}</td><td>${time}</td><td>${esc(h.player_cards)}</td><td>${esc(h.dealer_card)}</td>
          <td>${h.true_count != null ? Number(h.true_count).toFixed(1) : ""}</td>
          <td>${esc(ACTION_LABELS[h.recommended_action]?.[0] ?? h.recommended_action ?? "")}</td>
          <td class="${dev ? "dev" : ""}" title="${dev ? "Distinta de la recomendada" : ""}">${esc(ACTION_LABELS[h.action_taken]?.[0] ?? h.action_taken)}</td>
          <td>${esc(RESULT_LABELS[h.result] ?? h.result)}</td>
          <td class="${net > 0 ? "pos" : net < 0 ? "neg" : ""}">${net > 0 ? "+" : ""}${net}</td>
          <td><button title="Borrar" data-id="${h.id}">×</button></td>`;
        tr.querySelector("button").addEventListener("click", async () => {
          await api(`/history/${h.id}`, undefined, "DELETE").catch((e) => toast(e.message));
          loadHistory();
        });
        return tr;
      })
    );
    const followed = stats.hands ? stats.followed / stats.hands : 0;
    $("#history-summary").textContent = stats.hands
      ? `${stats.hands} manos · neto ${stats.net_units > 0 ? "+" : ""}${Number(stats.net_units).toFixed(1)} u · ${pct(followed, 0)} según recomendación`
      : "sin manos guardadas";
  } catch (e) {
    toast(e.message);
  }
}

// ---------------------------------------------------------------- Reglas
async function loadSettings() {
  const { rules } = await api("/state");
  const f = $("#settings-form");
  f.num_decks.value = rules.num_decks;
  f.enhc.checked = rules.enhc;
  f.dealer_hits_soft17.checked = rules.dealer_hits_soft17;
  f.double_after_split.checked = rules.double_after_split;
  f.surrender.checked = rules.surrender;
  f.blackjack_payout.value = String(rules.blackjack_payout);
}

async function applySettings() {
  const f = $("#settings-form");
  try {
    await api("/reset", {
      num_decks: Number(f.num_decks.value),
      enhc: f.enhc.checked,
      dealer_hits_soft17: f.dealer_hits_soft17.checked,
      double_after_split: f.double_after_split.checked,
      surrender: f.surrender.checked,
      blackjack_payout: Number(f.blackjack_payout.value),
    });
    store.set("configured", true);
    clearRound();
    toast("Reglas aplicadas · zapato nuevo", false);
  } catch (e) {
    toast(e.message);
  }
}

async function openSettings() {
  await loadSettings().catch(() => {});
  $("#settings").showModal();
}

// ---------------------------------------------------------------- Init
function init() {
  const pad = $("#keypad");
  for (const r of RANKS) {
    const b = document.createElement("button");
    b.dataset.card = r;
    b.innerHTML = `${r}${KEY_HINT[r] ? `<span class="key">${KEY_HINT[r]}</span>` : ""}`;
    // pointerdown en vez de click: registra la carta sin esperar a soltar el dedo
    b.addEventListener("pointerdown", (e) => { e.preventDefault(); addCard(r); });
    b.addEventListener("keydown", (e) => { if (e.key === " ") e.preventDefault(); });
    pad.appendChild(b);
  }

  document.querySelectorAll(".zone").forEach((el) => el.addEventListener("click", () => setZone(el.dataset.zone)));
  $("#btn-undo").addEventListener("click", undo);
  $("#btn-clear").addEventListener("click", clearRound);
  $("#btn-new-round").addEventListener("click", newRound);
  $("#btn-undo-round").addEventListener("click", undoRound);
  $("#btn-save").addEventListener("click", openSave);
  $("#btn-exact-adv").addEventListener("click", exactAdvantage);

  // Preferencias persistentes
  const mode = $("#engine-mode");
  mode.value = store.get("mode", "auto");
  mode.addEventListener("change", () => { store.set("mode", mode.value); mode.blur(); refresh(); });
  const kelly = { "#k-bankroll": ["bankroll", ""], "#k-fraction": ["fraction", "0.5"], "#k-min": ["minBet", "10"], "#k-max": ["maxBet", ""] };
  for (const [sel, [key, def]] of Object.entries(kelly)) {
    const el = $(sel);
    el.value = store.get(key, def);
    el.addEventListener("change", () => { store.set(key, el.value); refresh(); });
  }
  const openPanels = store.get("panels", {});
  document.querySelectorAll("details.panel").forEach((d) => {
    if (d.dataset.key in openPanels) d.open = openPanels[d.dataset.key];
    d.addEventListener("toggle", () => {
      const cur = store.get("panels", {});
      cur[d.dataset.key] = d.open;
      store.set("panels", cur);
      if (d.id === "history-panel" && d.open) loadHistory();
    });
  });

  // Diálogo de reglas
  const dlg = $("#settings");
  $("#btn-settings").addEventListener("click", openSettings);
  dlg.addEventListener("close", () => {
    if (dlg.returnValue === "apply") applySettings();
    else store.set("configured", true);
  });

  // Diálogo de guardar
  const sdlg = $("#save-dialog");
  document.querySelectorAll("#seg-action button").forEach((b) => b.addEventListener("click", () => selectSaveAction(b.dataset.v)));
  document.querySelectorAll("#seg-result button").forEach((b) => b.addEventListener("click", () => selectSaveResult(b.dataset.v)));
  $("#save-net").addEventListener("input", () => { saveState.netEdited = true; });
  $("#save-form").addEventListener("submit", (e) => {
    if (e.submitter && e.submitter.value === "cancel") return;
    e.preventDefault();
    confirmSave();
  });
  sdlg.addEventListener("keydown", (e) => {
    const inInput = e.target.tagName === "INPUT";
    const key = e.key.toLowerCase();
    if (key === "enter") { e.preventDefault(); confirmSave(); return; }
    if (inInput) return;
    if (SAVE_ACTION_KEYS[key]) { e.preventDefault(); selectSaveAction(SAVE_ACTION_KEYS[key]); }
    else if (SAVE_RESULT_KEYS[key]) { e.preventDefault(); selectSaveResult(SAVE_RESULT_KEYS[key]); }
  });

  // Atajos globales de registro rápido
  document.addEventListener("keydown", (e) => {
    if (dlg.open || sdlg.open || e.ctrlKey || e.metaKey || e.altKey) return;
    const tag = e.target.tagName;
    if (tag === "INPUT" || tag === "SELECT") return;
    const key = e.key.toLowerCase();
    if (KEY_MAP[key]) { e.preventDefault(); addCard(KEY_MAP[key]); return; }
    switch (key) {
      case "m": setZone("player"); break;
      case "d": setZone("dealer"); break;
      case "x": setZone("table"); break;
      case "g": e.preventDefault(); openSave(); break;
      case "tab": {
        e.preventDefault();
        const i = ZONES.indexOf(state.zone);
        setZone(ZONES[(i + (e.shiftKey ? ZONES.length - 1 : 1)) % ZONES.length]);
        break;
      }
      case "backspace": e.preventDefault(); undo(); break;
      case "escape": clearRound(); break;
      case "enter": e.preventDefault(); newRound(); break;
      default: return;
    }
  });

  setZone("player");
  refresh();
  if ($("#history-panel").open) loadHistory();
  if (!store.get("configured", false)) openSettings(); // configuración inicial de la mesa
}

init();
