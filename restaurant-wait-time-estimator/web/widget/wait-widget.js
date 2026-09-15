/**
 * <wte-wait-widget> — embeddable wait-time widget (blueprint §8.2 / §8.3).
 *
 * Usage:
 *   <script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.5.1/echarts.min.js"></script>
 *   <script src="wait-widget.js"></script>
 *   <wte-wait-widget api="https://api.example.com" token="…" location="lisboa-baixa"></wte-wait-widget>
 *
 * Shows: band label ("10–15 min"), colour state, and the "Best time today" chart with
 * q50 line + q10–q90 band + live observations. Live updates via WebSocket, REST fallback.
 */
(function () {
  const TEMPLATE = `
    <style>
      :host { display:block; font-family: system-ui, -apple-system, sans-serif; color:#111; }
      .card { border-radius:16px; padding:16px; background:#fff; box-shadow:0 2px 12px rgba(0,0,0,.08); max-width:420px; }
      .row { display:flex; align-items:baseline; gap:12px; }
      .label { font-size:2.2rem; font-weight:700; }
      .pill { font-size:.8rem; padding:4px 10px; border-radius:999px; color:#fff; }
      .short { background:#16a34a } .normal { background:#f59e0b } .long { background:#dc2626 } .na { background:#6b7280 }
      .meta { font-size:.8rem; color:#555; margin-top:4px }
      .chart { height:200px; margin-top:12px }
      .best { font-size:.85rem; margin-top:8px }
      .best b { color:#16a34a }
    </style>
    <div class="card">
      <div class="row"><span class="label" id="label">…</span><span class="pill na" id="pill">loading</span></div>
      <div class="meta" id="meta"></div>
      <div class="chart" id="chart"></div>
      <div class="best" id="best"></div>
    </div>`;

  class WteWaitWidget extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: "open" }).innerHTML = TEMPLATE;
      this.$ = (id) => this.shadowRoot.getElementById(id);
    }

    connectedCallback() {
      this.api = this.getAttribute("api") || "";
      this.token = this.getAttribute("token") || "";
      this.location = this.getAttribute("location") || "";
      this.live = [];
      this.fetchWait();
      this.fetchForecast();
      this.openSocket();
      this.timer = setInterval(() => this.fetchWait(), 30000);
    }

    disconnectedCallback() {
      clearInterval(this.timer);
      if (this.ws) this.ws.close();
    }

    headers() { return { authorization: "Bearer " + this.token }; }

    async fetchWait() {
      try {
        const r = await fetch(`${this.api}/v1/locations/${this.location}/wait`, { headers: this.headers() });
        if (r.ok) this.render(await r.json());
      } catch (e) { /* offline: keep last state */ }
    }

    async fetchForecast() {
      try {
        const r = await fetch(`${this.api}/v1/locations/${this.location}/forecast?hours=24`, { headers: this.headers() });
        if (r.ok) this.renderChart(await r.json());
        const b = await fetch(`${this.api}/v1/locations/${this.location}/best-times?n=3`, { headers: this.headers() });
        if (b.ok) this.renderBest(await b.json());
      } catch (e) { /* no forecast yet */ }
    }

    openSocket() {
      try {
        const url = this.api.replace(/^http/, "ws") + `/v1/ws/locations/${this.location}?token=${encodeURIComponent(this.token)}`;
        this.ws = new WebSocket(url);
        this.ws.onmessage = (m) => {
          const d = JSON.parse(m.data);
          if (d.type === "wait_estimate") this.render({ estimate: d, label: d.label });
        };
        this.ws.onclose = () => setTimeout(() => this.openSocket(), 5000);
      } catch (e) { /* REST polling continues */ }
    }

    render(data) {
      const est = data.estimate;
      const label = data.label || "n/a";
      this.$("label").textContent = label === "n/a" ? "—" : label;
      const p90 = est.p90_minutes;
      const cls = p90 == null ? "na" : p90 < 5 ? "short" : p90 < 15 ? "normal" : "long";
      const pill = this.$("pill");
      pill.className = "pill " + cls;
      pill.textContent = p90 == null ? "no data" : cls === "short" ? "walk in" : cls === "normal" ? "short wait" : "busy";
      const src = est.source === "pos_only" ? "estimated from orders (camera offline)" : est.source === "vision" ? "live camera analytics" : "";
      this.$("meta").textContent = `${est.queue_length} in queue · ${src}`;
      if (est.p50_minutes != null) {
        this.live.push([new Date(est.ts), est.p50_minutes]);
        if (this.live.length > 200) this.live.shift();
        if (this.chart) this.chart.setOption({ series: [{ id: "live", data: this.live }] });
      }
    }

    renderChart(fc) {
      if (typeof echarts === "undefined") return;
      this.chart = this.chart || echarts.init(this.$("chart"));
      const ts = fc.points.map((p) => new Date(p.ts));
      const q50 = fc.points.map((p, i) => [ts[i], +p.q50.toFixed(1)]);
      const q10 = fc.points.map((p, i) => [ts[i], +p.q10.toFixed(1)]);
      const band = fc.points.map((p, i) => [ts[i], +(p.q90 - p.q10).toFixed(1)]);
      this.chart.setOption({
        grid: { left: 36, right: 12, top: 12, bottom: 24 },
        tooltip: { trigger: "axis", valueFormatter: (v) => `${v} min` },
        xAxis: { type: "time" },
        yAxis: { type: "value", name: "min", nameGap: 8 },
        dataZoom: [{ type: "inside" }],
        series: [
          { id: "q10", type: "line", data: q10, stack: "band", lineStyle: { opacity: 0 }, symbol: "none", silent: true },
          { id: "band", type: "line", data: band, stack: "band", areaStyle: { color: "rgba(59,130,246,.18)" }, lineStyle: { opacity: 0 }, symbol: "none", name: "q10–q90" },
          { id: "q50", type: "line", data: q50, name: "expected", smooth: true, symbol: "none", lineStyle: { width: 3, color: "#2563eb" } },
          { id: "live", type: "line", data: this.live, name: "today (live)", symbol: "circle", symbolSize: 4, lineStyle: { type: "dotted", color: "#111" } },
        ],
      });
    }

    renderBest(list) {
      if (!list.length) return;
      const fmt = (s) => new Date(s).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      this.$("best").innerHTML = "Best times today: " + list.map((b) => `<b>${fmt(b.ts)}</b> (~${Math.round(b.q50)} min)`).join(" · ");
    }
  }

  customElements.define("wte-wait-widget", WteWaitWidget);
})();
