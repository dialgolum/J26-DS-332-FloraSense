import { useEffect, useMemo, useState } from "react";
import {
  Activity, AlertTriangle, ArrowDownRight, ArrowUpRight, BadgeCheck,
  CheckCircle2, CloudRain, Info, Leaf, Loader2, PackageCheck,
  Sprout, Thermometer, TrendingUp,
} from "lucide-react";
import apiClient from "../../api/apiClient.js";
import Button from "../ui/Button.jsx";
import Card from "../ui/Card.jsx";
import ErrorMessage from "../ui/ErrorMessage.jsx";
import PageIntro from "../ui/PageIntro.jsx";
import StatusBadge from "../ui/StatusBadge.jsx";

const API_ROOT = "http://127.0.0.1:8000";
const FALLBACKS = {
  customers: ["CUSTOMER-A", "CUSTOMER-B", "CUSTOMER-C", "CUSTOMER-D", "CUSTOMER-E", "CUSTOMER-F"],
  plant_varieties: ["Aglaonema commutatum", "Codiaeum spp.", "Cordyline fruticosa", "Dracaena sanderiana"],
  sizes: ["10/12cm", "15/17cm", "20/25cm", "Top-Cutting"],
  modes: ["Un-rooted", "Rooted"],
};

const numberFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const decimalFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

function unwrapResponse(responseData) {
  return responseData?.data?.data ?? responseData?.data ?? responseData ?? {};
}

function isNetworkFailure(error) {
  return !error?.response && (error?.request || error?.code === "ERR_NETWORK");
}

function Field({ id, label, hint, children }) {
  return (
    <div>
      <label htmlFor={id} className="mb-2 block text-sm font-bold text-ink">{label}</label>
      {children}
      {hint && <p className="mt-1.5 text-xs leading-5 text-muted">{hint}</p>}
    </div>
  );
}

function NumberInput({ id, value, onChange, min, max, step = 1, suffix }) {
  return (
    <div className="relative">
      <input
        id={id}
        className="field-control pr-14"
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(event.target.value)}
        required
      />
      {suffix && <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-xs font-semibold text-muted">{suffix}</span>}
    </div>
  );
}

function SelectInput({ id, value, onChange, options, loading }) {
  return (
    <select id={id} className="field-control" value={value} onChange={(event) => onChange(event.target.value)} disabled={loading}>
      {options.map((option) => <option key={option} value={option}>{option}</option>)}
    </select>
  );
}

function KpiCard({ title, value, unit, icon: Icon, tone = "neutral", detail }) {
  const tones = {
    neutral: "bg-[#f7f9f6] text-forest",
    positive: "bg-emerald-50 text-emerald-800",
    negative: "bg-rose-50 text-rose-800",
    mint: "bg-mint text-leaf",
  };
  return (
    <div className="rounded-2xl border border-line bg-white p-4 sm:p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-bold uppercase tracking-[.1em] text-muted">{title}</p>
          <p className="mt-3 font-display text-2xl font-extrabold tracking-tight text-forest sm:text-3xl">{value}</p>
          {unit && <p className="mt-0.5 text-xs font-semibold text-muted">{unit}</p>}
        </div>
        <span className={`grid h-10 w-10 shrink-0 place-items-center rounded-xl ${tones[tone]}`}><Icon size={19} /></span>
      </div>
      {detail && <p className="mt-3 text-xs leading-5 text-muted">{detail}</p>}
    </div>
  );
}

function statusForGap(gap) {
  if (gap < 0) return { label: "Deficit / Risk", tone: "fail", color: "rose" };
  if (gap > 0) return { label: "Surplus", tone: "ready", color: "emerald" };
  return { label: "Balanced", tone: "neutral", color: "slate" };
}

function Advisory({ forecast, gap, rate }) {
  if (gap < 0) {
    return (
      <div className="flex gap-3 rounded-2xl border border-rose-200 bg-rose-50 p-4 sm:p-5" role="status">
        <AlertTriangle className="mt-0.5 shrink-0 text-rose-700" size={20} />
        <div>
          <p className="font-bold text-rose-900">Fulfillment deficit warning</p>
          <p className="mt-1 text-sm leading-6 text-rose-800">
            Projected dispatch is short by <strong>{numberFormat.format(Math.abs(gap))} stems</strong>. Review the climate-lag conditions and consider allocating reserve rooting beds or negotiating a partial consignment dispatch.
          </p>
        </div>
      </div>
    );
  }
  if (gap > 0) {
    return (
      <div className="flex gap-3 rounded-2xl border border-emerald-200 bg-emerald-50 p-4 sm:p-5" role="status">
        <CheckCircle2 className="mt-0.5 shrink-0 text-emerald-700" size={20} />
        <div>
          <p className="font-bold text-emerald-900">Export target achievable</p>
          <p className="mt-1 text-sm leading-6 text-emerald-800">
            Projected harvest exceeds buyer contract demand by <strong>{numberFormat.format(gap)} stems</strong>, with an estimated fulfillment rate of <strong>{decimalFormat.format(rate)}%</strong>.
          </p>
        </div>
      </div>
    );
  }
  return (
    <div className="flex gap-3 rounded-2xl border border-line bg-[#f7f9f6] p-4 sm:p-5" role="status">
      <PackageCheck className="mt-0.5 shrink-0 text-forest" size={20} />
      <div>
        <p className="font-bold text-forest">Demand and projected dispatch are balanced</p>
        <p className="mt-1 text-sm leading-6 text-muted">
          The forecast matches committed demand. Continue normal packhouse allocation and monitor the shipment week.
        </p>
      </div>
    </div>
  );
}

export default function C4SupplyDemandForecast() {
  const currentYear = new Date().getFullYear();
  const [metadata, setMetadata] = useState(FALLBACKS);
  const [metadataLoading, setMetadataLoading] = useState(true);
  const [backendUnavailable, setBackendUnavailable] = useState(false);
  const [metadataNotice, setMetadataNotice] = useState("");
  const [form, setForm] = useState({
    customer: FALLBACKS.customers[0],
    plant_variety: FALLBACKS.plant_varieties[0],
    size: FALLBACKS.sizes[1],
    mode: FALLBACKS.modes[0],
    ordered_quantity: "10000",
    year: String(currentYear),
    week: "40",
    avg_max_temp_c: "33.5",
    avg_min_temp_c: "24.0",
    weekly_rainfall_mm: "15.0",
  });
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(false);
  const [requestError, setRequestError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    apiClient.get("/api/v1/forecast/metadata", { signal: controller.signal })
      .then((response) => {
        if (!active) return;
        const data = unwrapResponse(response.data);
        const nextMetadata = Object.fromEntries(
          Object.keys(FALLBACKS).map((key) => [
            key,
            Array.isArray(data[key]) && data[key].length ? data[key] : FALLBACKS[key],
          ]),
        );
        setMetadata(nextMetadata);
        setBackendUnavailable(false);
        setMetadataNotice(Object.keys(FALLBACKS).some((key) => !Array.isArray(data[key]) || !data[key].length)
          ? "Some metadata lists were unavailable; fallback options are shown."
          : "");
        setForm((previous) => Object.fromEntries(Object.entries(previous).map(([key, value]) => {
          const metadataKey = {
            customer: "customers", plant_variety: "plant_varieties", size: "sizes", mode: "modes",
          }[key];
          return Array.isArray(nextMetadata[metadataKey]) && !nextMetadata[metadataKey].includes(value)
            ? [key, nextMetadata[metadataKey][0]]
            : [key, value];
        })));
      })
      .catch((error) => {
        if (!active || error.code === "ERR_CANCELED") return;
        if (isNetworkFailure(error)) setBackendUnavailable(true);
        else setMetadataNotice("Metadata could not be loaded; fallback options are shown.");
      })
      .finally(() => { if (active) setMetadataLoading(false); });

    return () => { active = false; controller.abort(); };
  }, []);

  const update = (key) => (value) => setForm((previous) => ({ ...previous, [key]: value }));
  const gap = forecast ? Number(forecast.supply_demand_gap) : null;
  const dispatch = forecast ? Number(forecast.predicted_dispatched_quantity) : null;
  const demand = forecast ? Number(forecast.ordered_quantity) : Number(form.ordered_quantity);
  const calculatedRate = forecast && Number.isFinite(dispatch) && demand > 0 ? (dispatch / demand) * 100 : null;
  const fulfillmentRate = forecast && Number.isFinite(Number(forecast.fulfillment_rate_percent))
    ? Number(forecast.fulfillment_rate_percent)
    : calculatedRate;
  const risk = Number.isFinite(gap) ? statusForGap(gap) : null;
  const rateProgress = useMemo(() => Math.max(0, Math.min(100, fulfillmentRate || 0)), [fulfillmentRate]);

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setRequestError("");
    setForecast(null);
    const payload = {
      customer: form.customer,
      plant_variety: form.plant_variety,
      size: form.size,
      mode: form.mode,
      ordered_quantity: Number(form.ordered_quantity),
      year: Number(form.year),
      week: Number(form.week),
      avg_max_temp_c: Number(form.avg_max_temp_c),
      avg_min_temp_c: Number(form.avg_min_temp_c),
      weekly_rainfall_mm: Number(form.weekly_rainfall_mm),
    };
    try {
      const response = await apiClient.post("/api/v1/forecast/predict", payload);
      const data = unwrapResponse(response.data);
      const predicted = Number(data.predicted_dispatched_quantity);
      const ordered = Number(data.ordered_quantity ?? payload.ordered_quantity);
      const supplyGap = Number.isFinite(Number(data.supply_demand_gap))
        ? Number(data.supply_demand_gap)
        : predicted - ordered;
      if (!Number.isFinite(predicted)) throw new Error("The forecast response did not include a valid predicted dispatch quantity.");
      setForecast({ ...data, ...payload, ordered_quantity: ordered, predicted_dispatched_quantity: predicted, supply_demand_gap: supplyGap });
      setBackendUnavailable(false);
    } catch (error) {
      if (isNetworkFailure(error)) {
        setBackendUnavailable(true);
        setRequestError("");
      } else {
        setRequestError(error.response?.data?.detail || error.message || "The forecast request could not be completed.");
      }
    } finally {
      setLoading(false);
    }
  }

  const kpiGapTone = risk?.color === "rose" ? "negative" : risk?.color === "emerald" ? "positive" : "neutral";

  return (
    <>
      <PageIntro
        eyebrow="Commercial Operations · Component 4"
        title="FloraSense — Export Supply-Demand Planning Portal"
        description="Real-time harvest throughput projections and contract fulfillment reconciliation conditioned on ambient microclimate."
        action={<StatusBadge tone={backendUnavailable ? "review" : "ready"} dot>{backendUnavailable ? "Service Offline" : "Live Operational Feed"}</StatusBadge>}
      />

      {backendUnavailable && (
        <div role="alert" className="mb-5 flex gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950">
          <AlertTriangle size={18} className="mt-0.5 shrink-0 text-amber-700" />
          <div>
            <p className="font-bold">FastAPI backend unreachable</p>
            <p className="mt-1 leading-5">FastAPI backend unreachable at {API_ROOT}. Start it from the repository root with <code className="rounded bg-amber-100 px-1 py-0.5">python -m uvicorn services.c4_forecasting.api.main:app --reload --host 127.0.0.1 --port 8000</code>.</p>
          </div>
        </div>
      )}
      {metadataNotice && !backendUnavailable && <div role="status" className="mb-5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">{metadataNotice}</div>}
      {requestError && <div className="mb-5"><ErrorMessage title="Forecast request failed">{requestError}</ErrorMessage></div>}

      <div className="grid items-start gap-5 xl:grid-cols-[.88fr_1.12fr]">
        <Card className="overflow-hidden">
          <div className="border-b border-line bg-[#fbfcfa] px-5 py-5 sm:px-6">
            <div className="flex items-center gap-3">
              <span className="grid h-10 w-10 place-items-center rounded-xl bg-mint text-leaf"><Leaf size={19} /></span>
              <div>
                <h3 className="font-display text-lg font-extrabold text-forest">Forecast parameters</h3>
                <p className="mt-0.5 text-xs text-muted">Set contract, shipment timing, and expected climate conditions.</p>
              </div>
            </div>
          </div>
          <form onSubmit={handleSubmit} className="space-y-5 p-5 sm:p-6">
            <div className="grid gap-4 sm:grid-cols-2">
              <Field id="c4-customer" label="Customer">
                <SelectInput id="c4-customer" value={form.customer} onChange={update("customer")} options={metadata.customers} loading={metadataLoading} />
              </Field>
              <Field id="c4-mode" label="Root mode">
                <SelectInput id="c4-mode" value={form.mode} onChange={update("mode")} options={metadata.modes} loading={metadataLoading} />
              </Field>
              <Field id="c4-variety" label="Plant variety">
                <SelectInput id="c4-variety" value={form.plant_variety} onChange={update("plant_variety")} options={metadata.plant_varieties} loading={metadataLoading} />
              </Field>
              <Field id="c4-size" label="Plant size">
                <SelectInput id="c4-size" value={form.size} onChange={update("size")} options={metadata.sizes} loading={metadataLoading} />
              </Field>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field id="c4-week" label="Target shipment week" hint="ISO-style week number used by the forecast service.">
                <NumberInput id="c4-week" value={form.week} onChange={update("week")} min="1" max="52" suffix="week" />
              </Field>
              <Field id="c4-year" label="Forecast year">
                <NumberInput id="c4-year" value={form.year} onChange={update("year")} min="2023" max="2100" suffix="year" />
              </Field>
            </div>

            <Field id="c4-demand" label="Buyer Contract Demand" hint="Required for this post-order dispatch supply forecast.">
              <NumberInput id="c4-demand" value={form.ordered_quantity} onChange={update("ordered_quantity")} min="0" suffix="stems" />
            </Field>

            <div className="rounded-2xl border border-line bg-[#fbfcfa] p-4 sm:p-5">
              <div className="mb-4 flex items-start justify-between gap-3">
                <div>
                  <p className="font-display font-extrabold text-forest">Nursery Microclimate Forecast</p>
                  <p className="mt-1 text-xs text-muted">Enter expected conditions for the shipment week.</p>
                </div>
                <span title="The forecasting service uses its 2-week and 4-week weather-lag features to represent delayed crop response. This form sends the expected weekly conditions; lag construction is handled by the backend model pipeline." aria-label="About weather lags" className="grid h-8 w-8 shrink-0 cursor-help place-items-center rounded-lg bg-white text-leaf ring-1 ring-line"><Info size={16} /></span>
              </div>
              <div className="grid gap-4 sm:grid-cols-3">
                <Field id="c4-max-temp" label="Expected max temp" hint="Nursery preset: 33.5°C">
                  <NumberInput id="c4-max-temp" value={form.avg_max_temp_c} onChange={update("avg_max_temp_c")} min="-10" max="60" step="0.1" suffix="°C" />
                </Field>
                <Field id="c4-min-temp" label="Expected min temp" hint="Nursery preset: 24.0°C">
                  <NumberInput id="c4-min-temp" value={form.avg_min_temp_c} onChange={update("avg_min_temp_c")} min="-10" max="60" step="0.1" suffix="°C" />
                </Field>
                <Field id="c4-rain" label="Weekly rainfall" hint="Nursery preset: 15.0 mm">
                  <NumberInput id="c4-rain" value={form.weekly_rainfall_mm} onChange={update("weekly_rainfall_mm")} min="0" max="2000" step="0.1" suffix="mm" />
                </Field>
              </div>
              <div className="mt-4 flex flex-wrap gap-2 text-[10px] font-semibold text-muted">
                <span className="inline-flex items-center gap-1 rounded-full bg-white px-2.5 py-1 ring-1 ring-line"><Thermometer size={12} /> Temperature</span>
                <span className="inline-flex items-center gap-1 rounded-full bg-white px-2.5 py-1 ring-1 ring-line"><CloudRain size={12} /> Precipitation</span>
                <span className="inline-flex items-center gap-1 rounded-full bg-white px-2.5 py-1 ring-1 ring-line"><TrendingUp size={12} /> 2W / 4W lag features</span>
              </div>
            </div>

            <Button type="submit" disabled={loading || metadataLoading} className="w-full shadow-sm">
              {loading ? <><Loader2 size={17} className="animate-spin" />Calculating forecast…</> : <><Activity size={17} />Run Dispatch Feasibility Analysis</>}
            </Button>
            <p className="text-center text-[11px] leading-5 text-muted">Forecasts support packhouse planning and should be reviewed against current crop and order conditions.</p>
          </form>
        </Card>

        <div className="space-y-5">
          <Card className="overflow-hidden">
            <div className="flex flex-col justify-between gap-3 border-b border-line bg-[#fbfcfa] px-5 py-5 sm:flex-row sm:items-center sm:px-6">
              <div>
                <p className="text-[10px] font-bold uppercase tracking-[.15em] text-leaf">Manager view · {forecast ? `${forecast.year} / week ${forecast.week}` : `${form.year} / week ${form.week}`}</p>
                <h3 className="mt-1 font-display text-lg font-extrabold text-forest">Live forecast output</h3>
              </div>
              {risk && <StatusBadge tone={risk.tone} dot>{risk.label}</StatusBadge>}
            </div>
            <div className="space-y-4 p-5 sm:p-6">
              <div className="grid gap-3 sm:grid-cols-2">
                <KpiCard title="Buyer Contract Demand" value={numberFormat.format(demand || 0)} unit="stems" icon={PackageCheck} tone="mint" detail="Contract quantity to fulfill" />
                <KpiCard title="Projected Packhouse Yield" value={dispatch == null ? "—" : numberFormat.format(dispatch)} unit="stems" icon={Sprout} tone="neutral" detail={forecast ? "Model-estimated harvest dispatch" : "Run a forecast to estimate available supply"} />
                <KpiCard title="Net Export Balance" value={gap == null ? "—" : `${gap > 0 ? "+" : ""}${numberFormat.format(gap)}`} unit="stems" icon={gap != null && gap < 0 ? ArrowDownRight : ArrowUpRight} tone={gap == null ? "neutral" : kpiGapTone} detail={risk ? risk.label : "Positive = surplus · negative = deficit"} />
                
                <div className="rounded-2xl border border-line bg-white p-4 sm:p-5">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-xs font-bold uppercase tracking-[.1em] text-muted">Fulfillment rate</p>
                      <p className="mt-3 font-display text-2xl font-extrabold tracking-tight text-forest sm:text-3xl">{fulfillmentRate == null ? "—" : `${decimalFormat.format(fulfillmentRate)}%`}</p>
                      <p className="mt-0.5 text-xs font-semibold text-muted">dispatch supply ÷ order demand</p>
                    </div>
                    <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-mint text-leaf"><BadgeCheck size={19} /></span>
                  </div>
                  <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-100" role="progressbar" aria-label="Fulfillment rate" aria-valuemin="0" aria-valuemax="100" aria-valuenow={rateProgress}>
                    <div className={`h-full rounded-full transition-all ${gap != null && gap < 0 ? "bg-rose-500" : "bg-emerald-600"}`} style={{ width: `${rateProgress}%` }} />
                  </div>
                </div>
              </div>

              {forecast ? (
                <Advisory forecast={forecast} gap={gap} rate={fulfillmentRate ?? 0} />
              ) : (
                <div className="grid min-h-28 place-items-center rounded-2xl border border-dashed border-line bg-[#fbfcfa] px-5 py-6 text-center">
                  <div>
                    <span className="mx-auto grid h-10 w-10 place-items-center rounded-xl bg-mint text-leaf"><Activity size={19} /></span>
                    <p className="mt-2 text-sm font-bold text-forest">Forecast results will appear here</p>
                    <p className="mt-1 text-xs text-muted">Submit an order and shipment scenario to calculate the supply gap.</p>
                  </div>
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </>
  );
}