import {
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useState,
  type ComponentType,
  type ReactNode,
} from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownToLine,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Database,
  Droplets,
  FileText,
  FlaskConical,
  MapPinned,
  Search,
} from "lucide-react";
import "./App.css";

const API = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8001/api";
const MapView = lazy(() => import("./SpatialMap"));
const TemporalChart = lazy(() => import("./TemporalChart"));
type Page =
  | "Dashboard"
  | "Dataset"
  | "Spatial Analysis"
  | "Temporal Analysis"
  | "WQI Prediction"
  | "Reports";
type Summary = {
  record_count: number;
  village_count: number;
  block_count: number;
  year_count: number;
  year_range: [number, number] | null;
  source_quality_class_distribution: Record<string, number>;
  calculated_wqi_class_distribution: Record<string, number>;
  wqi_summary: {
    mean_wqi: number;
    minimum_wqi: number;
    maximum_wqi: number;
    included_parameters: string[];
    excluded_parameters: Record<string, string>;
    method_reference: string;
    standard_source: string;
    interpretation_note: string;
  };
  mean_observed_parameters: Record<
    string,
    { count: number; mean: number | null }
  >;
  wqi: { reason: string; method: string };
  models: { reason: string };
  preprocessing: {
    quarantined_malformed_records?: number;
    removed_exact_duplicates?: number;
  };
};
export type Point = {
  source_dataset: string;
  source_record_number: number;
  village: string;
  block: string;
  latitude: number;
  longitude: number;
  year: number;
  season: string;
  well_type: string;
  quality_class_source: string;
  risk_class_source: string;
  wqi: number;
  wqi_class: string;
  observation_status: "observed";
};
export type Spatial = {
  points: Point[];
  metadata: {
    input_records: number;
    unique_coordinate_records: number;
    selected_records: number;
    display_limit: number;
    is_limited: boolean;
    message: string;
    class_counts: Record<string, number>;
    display_class_counts: Record<string, number>;
    coordinate_bounds: {
      latitude_min: number;
      latitude_max: number;
      longitude_min: number;
      longitude_max: number;
    } | null;
    coordinate_note: string;
    available_filters: {
      years: number[];
      seasons: string[];
      blocks: string[];
      source_quality_classes: string[];
      wqi_classes: string[];
    };
  };
};
type ParamStats = { count: number; mean: number | null; median: number | null };
type Temporal = {
  observed_only: boolean;
  time_coverage: {
    first_year: number | null;
    last_year: number | null;
    years: number[];
    seasons: string[];
  };
  record_count_by_year: { year: number; record_count: number }[];
  annual_parameter_summary: {
    year: number;
    record_count: number;
    parameters: Record<string, ParamStats>;
  }[];
  year_season_parameter_summary: {
    year: number;
    season: string;
    record_count: number;
    parameters: Record<string, ParamStats>;
  }[];
  annual_source_class_distribution: {
    year: number;
    source_quality_label: string;
    record_count: number;
  }[];
  annual_wqi_class_distribution: {
    year: number;
    wqi_class: string;
    record_count: number;
  }[];
  limitations: string[];
};
type Dataset = {
  page: number;
  page_size: number;
  total_records: number;
  records: Record<string, string | number | null>[];
  filters: {
    years: number[];
    seasons: string[];
    blocks: string[];
    source_quality_classes: string[];
    wqi_classes: string[];
  };
};
type PredictionOptions = {
  feature_fields: string[];
  samples_are_illustrative: boolean;
  sample_profiles: {
    id: string;
    label: string;
    values: Record<string, number>;
  }[];
  guideline_source: string;
};
type Prediction = {
  prediction_type: "predicted";
  predicted_wqi: number;
  predicted_class: string;
  calculated_wqi: number | null;
  calculated_class: string | null;
  parameter_assessments: {
    field: string;
    label: string;
    value: number;
    unit: string;
    reference: string;
    status: string;
    tone: "good" | "caution" | "alert";
  }[];
};
const labels: Record<string, string> = {
  nitrate_mg_l: "Nitrate",
  fluoride_mg_l: "Fluoride",
  chloride_mg_l: "Chloride",
  tds_mg_l: "TDS",
  ph: "pH",
  iron_mg_l: "Iron",
  alkalinity_mg_l: "Alkalinity",
  wqi: "WQI",
};
const colors: Record<string, string> = {
  Excellent: "#27846d",
  Good: "#82a64b",
  Moderate: "#d5a63e",
  Poor: "#d17a3c",
  "Very Poor": "#ba584a",
  "Unsuitable for Drinking": "#a63d36",
  Unlabeled: "#89938c",
};
const nav: { title: Page; icon: ComponentType<{ size?: number }> }[] = [
  { title: "Dashboard", icon: Activity },
  { title: "Dataset", icon: Database },
  { title: "Spatial Analysis", icon: MapPinned },
  { title: "Temporal Analysis", icon: BarChart3 },
  { title: "WQI Prediction", icon: FlaskConical },
  { title: "Reports", icon: FileText },
];

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API}${path}`, { signal });
  if (!response.ok) throw new Error();
  return response.json() as Promise<T>;
}
async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    const detail =
      typeof errorBody === "object" && errorBody !== null && "detail" in errorBody
        ? errorBody.detail
        : null;
    throw new Error(
      typeof detail === "string"
        ? detail
        : `Prediction request failed (${response.status}).`,
    );
  }
  return response.json() as Promise<T>;
}
function number(value: number | null | undefined, digits = 1) {
  return value == null || !Number.isFinite(value)
    ? "—"
    : new Intl.NumberFormat("en-IN", { maximumFractionDigits: digits }).format(
        value,
      );
}
function color(label?: string | null) {
  return colors[label ?? ""] ?? "#89938c";
}

function MapPanel({
  data,
  compact = false,
}: {
  data: Spatial | null;
  compact?: boolean;
}) {
  return (
    <Suspense
      fallback={<div className="map-empty">Loading spatial layer…</div>}
    >
      <MapView data={data} compact={compact} />
    </Suspense>
  );
}
function Legend({ counts }: { counts: Record<string, number> }) {
  return (
    <div className="legend-row">
      {Object.entries(counts).map(([label, count]) => (
        <span className="legend-item" key={label}>
          <i style={{ background: color(label) }} />
          {label}
          <b>{count}</b>
        </span>
      ))}
    </div>
  );
}
function Heading({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {action}
    </div>
  );
}
function Loading() {
  return (
    <div className="loading-state">
      <span className="loading-ring" />
      <p>Loading observed groundwater records…</p>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("Dashboard"),
    [summary, setSummary] = useState<Summary | null>(null),
    [spatial, setSpatial] = useState<Spatial | null>(null),
    [temporal, setTemporal] = useState<Temporal | null>(null),
    [dataset, setDataset] = useState<Dataset | null>(null),
    [predictionOptions, setPredictionOptions] =
      useState<PredictionOptions | null>(null);
  const [error, setError] = useState(false),
    [loading, setLoading] = useState(true),
    [analyte, setAnalyte] = useState("nitrate_mg_l"),
    [pageNo, setPageNo] = useState(1),
    [search, setSearch] = useState(""),
    [year, setYear] = useState(""),
    [season, setSeason] = useState(""),
    [block, setBlock] = useState(""),
    [quality, setQuality] = useState(""),
    [wqiClass, setWqiClass] = useState("");
  useEffect(() => {
    const c = new AbortController();
    Promise.all([
      get<Summary>("/summary", c.signal),
      get<Spatial>("/spatial", c.signal),
    ])
      .then(([s, m]) => {
        setSummary(s);
        setSpatial(m);
        setError(false);
      })
      .catch((e: unknown) => {
        if (!(e instanceof DOMException && e.name === "AbortError"))
          setError(true);
      })
      .finally(() => setLoading(false));
    return () => c.abort();
  }, []);
  useEffect(() => {
    if (page !== "Temporal Analysis" || temporal) return;
    const c = new AbortController();
    get<Temporal>("/temporal", c.signal)
      .then(setTemporal)
      .catch((e: unknown) => {
        if (!(e instanceof DOMException && e.name === "AbortError"))
          setError(true);
      });
    return () => c.abort();
  }, [page, temporal]);
  useEffect(() => {
    if (page !== "Dataset") return;
    const c = new AbortController(),
      q = new URLSearchParams({ page: String(pageNo), page_size: "25" });
    if (search.trim()) q.set("search", search.trim());
    if (year) q.set("year", year);
    if (season) q.set("season", season);
    if (block) q.set("block", block);
    if (quality) q.set("quality_class", quality);
    if (wqiClass) q.set("wqi_class", wqiClass);
    get<Dataset>(`/dataset?${q}`, c.signal)
      .then((d) => {
        setDataset(d);
        setError(false);
      })
      .catch((e: unknown) => {
        if (!(e instanceof DOMException && e.name === "AbortError"))
          setError(true);
      });
    return () => c.abort();
  }, [page, pageNo, search, year, season, block, quality, wqiClass]);
  useEffect(() => {
    if (page !== "WQI Prediction" || predictionOptions) return;
    const c = new AbortController();
    get<PredictionOptions>("/prediction/options", c.signal)
      .then(setPredictionOptions)
      .catch((e: unknown) => {
        if (!(e instanceof DOMException && e.name === "AbortError"))
          setError(true);
      });
    return () => c.abort();
  }, [page, predictionOptions]);
  const chart = useMemo(
    () =>
      temporal?.annual_parameter_summary.map((r) => ({
        year: r.year,
        mean: r.parameters[analyte]?.mean ?? null,
      })) ?? [],
    [temporal, analyte],
  );
  async function downloadReport() {
    try {
      const data = await get<Record<string, unknown>>("/reports"),
        url = URL.createObjectURL(
          new Blob([JSON.stringify(data, null, 2)], {
            type: "application/json",
          }),
        ),
        link = document.createElement("a");
      link.href = url;
      link.download = "groundwater-observed-analysis-report.json";
      link.click();
      URL.revokeObjectURL(url);
    } catch {
      setError(true);
    }
  }
  const resetFilters = () => {
    setSearch("");
    setYear("");
    setSeason("");
    setBlock("");
    setQuality("");
    setWqiClass("");
    setPageNo(1);
  };
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#dashboard"
          onClick={() => setPage("Dashboard")}
        >
          <span className="brand-mark">
            <Droplets size={21} />
          </span>
          <span>
            <strong>Vellar</strong>
            <small>GROUNDWATER ATLAS</small>
          </span>
        </a>
        <div className="nav-heading">RESEARCH WORKSPACE</div>
        <nav>
          {nav.map(({ title, icon: Icon }) => (
            <button
              key={title}
              className={`nav-item${page === title ? " active" : ""}`}
              onClick={() => setPage(title)}
            >
              <Icon size={17} />
              <span>{title}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div>
            <i /> DATA CONNECTED
          </div>
          <span>2 source datasets · observed only</span>
          <span>Lower Vellar research workspace</span>
        </div>
      </aside>
      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            Groundwater intelligence <i>/</i> <strong>{page}</strong>
          </div>
          <div className="topbar-right">
            <span>
              <i /> Dataset ready
            </span>
            <span>Research preview</span>
          </div>
        </header>
        {error && (
          <div className="alert">
            <AlertTriangle size={17} />
            Unable to load groundwater data. Please try again.
            <button onClick={() => setError(false)}>×</button>
          </div>
        )}
        {loading && !summary ? (
          <Loading />
        ) : !summary ? (
          <div className="loading-state">
            Groundwater data unavailable. Please try again.
          </div>
        ) : (
          <>
            {page === "Dashboard" && (
              <Dashboard
                summary={summary}
                spatial={spatial}
                navigate={setPage}
              />
            )}
            {page === "Dataset" && (
              <DatasetPage
                data={dataset}
                search={search}
                setSearch={setSearch}
                year={year}
                setYear={setYear}
                season={season}
                setSeason={setSeason}
                block={block}
                setBlock={setBlock}
                quality={quality}
                setQuality={setQuality}
                wqiClass={wqiClass}
                setWqiClass={setWqiClass}
                pageNo={pageNo}
                setPageNo={setPageNo}
                reset={resetFilters}
              />
            )}
            {page === "Spatial Analysis" && <SpatialPage data={spatial} />}
            {page === "Temporal Analysis" && (
              <TemporalPage
                temporal={temporal}
                analyte={analyte}
                setAnalyte={setAnalyte}
                chart={chart}
              />
            )}
            {page === "WQI Prediction" && (
              <PredictionPage
                options={predictionOptions}
              />
            )}
            {page === "Reports" && (
              <Reports
                summary={summary}
                temporal={temporal}
                download={downloadReport}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}

function Metric({
  title,
  value,
  detail,
  icon: Icon,
  tone,
}: {
  title: string;
  value: string | number;
  detail: string;
  icon: ComponentType<{ size?: number }>;
  tone: string;
}) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span className={`metric-icon ${tone}`}>
          <Icon size={17} />
        </span>
        <span className="metric-tag">Observed</span>
      </div>
      <div className="metric-label">{title}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-detail">{detail}</div>
    </div>
  );
}
function Dashboard({
  summary,
  spatial,
  navigate,
}: {
  summary: Summary;
  spatial: Spatial | null;
  navigate: (p: Page) => void;
}) {
  return (
    <div className="page-view">
      <Heading
        eyebrow="LOWER VELLAR · OBSERVED DATA"
        title="Groundwater, in context."
        description="A field-led view of groundwater observations across Melbhuvanagiri and Kurinjipadi."
      />
      <div className="study-strip">
        <MapPinned size={16} />
        <span>
          <strong>Study area</strong> Cuddalore district, Tamil Nadu
        </span>
        <i />
        {summary.year_range?.join("–")}
        <span className="strip-right">4 seasonal rounds</span>
      </div>
      <div className="metric-grid">
        <Metric
          title="Observed records"
          value={number(summary.record_count, 0)}
          detail="Complete normalized observations"
          icon={Database}
          tone="green"
        />
        <Metric
          title="Villages represented"
          value={number(summary.village_count, 0)}
          detail={`${summary.block_count} source blocks`}
          icon={MapPinned}
          tone="blue"
        />
        <Metric
          title="Time span"
          value={summary.year_count}
          detail="Annual coverage"
          icon={Activity}
          tone="gold"
        />
        <div className="metric-card">
          <div className="metric-top">
            <span className="metric-icon pending">
              <FlaskConical size={17} />
            </span>
            <span className="metric-tag">Calculated</span>
          </div>
          <div className="metric-label">Mean calculated WQI</div>
          <div className="metric-value">
            {number(summary.wqi_summary.mean_wqi)}
          </div>
          <div className="metric-detail">
            Weighted arithmetic · six parameters
          </div>
        </div>
      </div>
      <div className="coordinate-warning">
        <AlertTriangle size={15} />
        <span>
          <strong>
            Verify source coordinates before geographic interpretation.
          </strong>{" "}
          Observed extent:{" "}
          {number(spatial?.metadata.coordinate_bounds?.latitude_min, 4)}–
          {number(spatial?.metadata.coordinate_bounds?.latitude_max, 4)}°
          latitude,{" "}
          {number(spatial?.metadata.coordinate_bounds?.longitude_min, 4)}–
          {number(spatial?.metadata.coordinate_bounds?.longitude_max, 4)}°
          longitude. The source has no boundary.
        </span>
      </div>
      <div className="dashboard-grid">
        <section className="panel map-panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">
                SPATIAL OVERVIEW · OBSERVED SAMPLES
              </div>
              <h2>Calculated WQI classes</h2>
            </div>
            <button
              className="text-button"
              onClick={() => navigate("Spatial Analysis")}
            >
              Explore map <ChevronRight size={15} />
            </button>
          </div>
          <MapPanel data={spatial} compact />
          <Legend counts={spatial?.metadata.display_class_counts ?? {}} />
          <div className="map-footnote">
            Map color is calculated WQI class; source label retained separately
            · coordinates not confirmed as Lower Vellar locations
          </div>
        </section>
        <section className="panel distribution-panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">CALCULATED WQI</div>
              <h2>Index class distribution</h2>
            </div>
          </div>
          <p className="panel-subtitle">
            Weighted arithmetic · source labels remain separate.
          </p>
          <div className="distribution-list">
            {Object.entries(summary.calculated_wqi_class_distribution)
              .sort((a, b) => b[1] - a[1])
              .map(([label, count]) => (
                <div className="distribution-row" key={label}>
                  <span>
                    <i style={{ background: color(label) }} />
                    {label}
                  </span>
                  <div className="distribution-track">
                    <i
                      style={{
                        width: `${(count / summary.record_count) * 100}%`,
                        background: color(label),
                      }}
                    />
                  </div>
                  <b>{count}</b>
                  <small>
                    {((count / summary.record_count) * 100).toFixed(1)}%
                  </small>
                </div>
              ))}
          </div>
          <div className="panel-divider" />
          <div className="small-stat">
            <span>Source labels marked Excellent</span>
            <b>{summary.source_quality_class_distribution.Excellent ?? 0}</b>
          </div>
          <div className="small-stat">
            <span>Calculated WQI above 100</span>
            <b>
              {(summary.calculated_wqi_class_distribution[
                "Unsuitable for Drinking"
              ] ?? 0) +
                (summary.calculated_wqi_class_distribution["Very Poor"] ?? 0)}
            </b>
          </div>
        </section>
      </div>
      <div className="bottom-grid">
        <section className="panel parameter-panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">OBSERVED CHEMISTRY</div>
              <h2>Dataset at a glance</h2>
            </div>
            <button className="text-button" onClick={() => navigate("Dataset")}>
              Browse records <ChevronRight size={15} />
            </button>
          </div>
          <div className="parameter-grid">
            {Object.entries(summary.mean_observed_parameters).map(
              ([key, value]) => (
                <div key={key}>
                  <span>{labels[key] ?? key}</span>
                  <b>{number(value.mean, key === "ph" ? 2 : 1)}</b>
                  <small>
                    {key === "ph" ? "unitless" : "mg/L"} mean · n={value.count}
                  </small>
                </div>
              ),
            )}
          </div>
        </section>
        <aside className="integrity-note">
          <AlertTriangle size={18} />
          <div>
            <strong>Index interpretation</strong>
            <p>
              Calculated WQI classes differ markedly from source labels; compare
              method and inputs before drawing conclusions.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}

function DatasetPage({
  data,
  search,
  setSearch,
  year,
  setYear,
  season,
  setSeason,
  block,
  setBlock,
  quality,
  setQuality,
  wqiClass,
  setWqiClass,
  pageNo,
  setPageNo,
  reset,
}: {
  data: Dataset | null;
  search: string;
  setSearch: (v: string) => void;
  year: string;
  setYear: (v: string) => void;
  season: string;
  setSeason: (v: string) => void;
  block: string;
  setBlock: (v: string) => void;
  quality: string;
  setQuality: (v: string) => void;
  wqiClass: string;
  setWqiClass: (v: string) => void;
  pageNo: number;
  setPageNo: (v: number) => void;
  reset: () => void;
}) {
  const totalPages = Math.max(1, Math.ceil((data?.total_records ?? 0) / 25)),
    cols = [
      ["village", "Location"],
      ["block", "Block"],
      ["year", "Year"],
      ["season", "Season"],
      ["well_type", "Source"],
      ["nitrate_mg_l", "Nitrate"],
      ["fluoride_mg_l", "Fluoride"],
      ["tds_mg_l", "TDS"],
      ["ph", "pH"],
      ["wqi", "WQI"],
      ["wqi_class", "WQI class"],
      ["quality_class_source", "Source label"],
    ] as const;
  return (
    <div className="page-view">
      <Heading
        eyebrow="SOURCE OBSERVATIONS"
        title="Dataset explorer"
        description="Search and filter the complete observed dataset. Results are paginated by the API."
      />
      <section className="panel dataset-panel">
        <div className="filter-toolbar">
          <label className="search-box">
            <Search size={16} />
            <input
              placeholder="Search village or block"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPageNo(1);
              }}
            />
          </label>
          <select
            aria-label="Year"
            value={year}
            onChange={(e) => {
              setYear(e.target.value);
              setPageNo(1);
            }}
          >
            <option value="">All years</option>
            {data?.filters.years.map((v) => <option key={v}>{v}</option>)}
          </select>
          <select
            aria-label="Season"
            value={season}
            onChange={(e) => {
              setSeason(e.target.value);
              setPageNo(1);
            }}
          >
            <option value="">All seasons</option>
            {data?.filters.seasons.map((v) => <option key={v}>{v}</option>)}
          </select>
          <select
            aria-label="Block"
            value={block}
            onChange={(e) => {
              setBlock(e.target.value);
              setPageNo(1);
            }}
          >
            <option value="">All blocks</option>
            {data?.filters.blocks.map((v) => <option key={v}>{v}</option>)}
          </select>
          <select
            aria-label="Calculated WQI class"
            value={wqiClass}
            onChange={(e) => {
              setWqiClass(e.target.value);
              setPageNo(1);
            }}
          >
            <option value="">All WQI classes</option>
            {data?.filters.wqi_classes.map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
          <select
            aria-label="Original source label"
            value={quality}
            onChange={(e) => {
              setQuality(e.target.value);
              setPageNo(1);
            }}
          >
            <option value="">All source labels</option>
            {data?.filters.source_quality_classes.map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
          <button className="clear-button" onClick={reset}>
            Clear
          </button>
        </div>
        <div className="table-meta">
          <span>
            {data
              ? `${data.total_records.toLocaleString()} matching records`
              : "Loading…"}
          </span>
          <span>
            {data?.records.length ?? 0} shown · page {pageNo} of {totalPages}
          </span>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                {cols.map(([, title]) => (
                  <th key={title}>{title}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data?.records.map((r) => (
                <tr key={`${r.source_dataset}-${r.source_record_number}`}>
                  {cols.map(([key]) => (
                    <td key={key}>
                      {key === "wqi_class" || key === "quality_class_source" ? (
                        <span
                          className="source-pill"
                          style={
                            {
                              "--class-color": color(String(r[key] ?? "")),
                            } as React.CSSProperties
                          }
                        >
                          {r[key] ?? "—"}
                        </span>
                      ) : key === "wqi" || key.endsWith("_mg_l") || key === "ph" ? (
                        number(Number(r[key]), 2)
                      ) : (
                        (r[key] ?? "—")
                      )}
                    </td>
                  ))}
                </tr>
              ))}
              {data && !data.records.length && (
                <tr>
                  <td colSpan={cols.length} className="empty-table">
                    No records match these filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="pagination">
          <span>Full dataset retained · calculated WQI and source label shown separately</span>
          <div>
            <button
              disabled={pageNo <= 1}
              onClick={() => setPageNo(pageNo - 1)}
              aria-label="Previous"
            >
              <ChevronLeft size={16} />
            </button>
            {pageNo} / {totalPages}
            <button
              disabled={pageNo >= totalPages}
              onClick={() => setPageNo(pageNo + 1)}
              aria-label="Next"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

function SpatialPage({ data }: { data: Spatial | null }) {
  const [view, setView] = useState(data),
    [mapYear, setMapYear] = useState(""),
    [mapSeason, setMapSeason] = useState(""),
    [mapBlock, setMapBlock] = useState(""),
    [mapClass, setMapClass] = useState("");
  useEffect(() => {
    if (!data) return;
    const controller = new AbortController(),
      query = new URLSearchParams();
    if (mapYear) query.set("year", mapYear);
    if (mapSeason) query.set("season", mapSeason);
    if (mapBlock) query.set("block", mapBlock);
    if (mapClass) query.set("wqi_class", mapClass);
    get<Spatial>(`/spatial?${query}`, controller.signal)
      .then(setView)
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError"))
          setView(null);
      });
    return () => controller.abort();
  }, [data, mapYear, mapSeason, mapBlock, mapClass]);
  const options = data?.metadata.available_filters;
  return (
    <div className="page-view">
      <Heading
        eyebrow="GIS · OBSERVED LOCATIONS"
        title="Spatial analysis"
        description="Observed samples colored by calculated WQI class. Confirm coordinates before interpreting geographic coverage."
      />
      <section className="panel spatial-panel">
        <div className="spatial-toolbar">
          <div>
            <div className="panel-kicker">
              OBSERVED MEASUREMENTS · CALCULATED WQI
            </div>
            <strong>{view?.metadata.message ?? "Map unavailable"}</strong>
          </div>
          <span className="observed-badge">
            <i /> Observed layer
          </span>
        </div>
        <div className="map-filters">
          <select
            aria-label="Map year"
            value={mapYear}
            onChange={(event) => setMapYear(event.target.value)}
          >
            <option value="">All years</option>
            {options?.years.map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
          <select
            aria-label="Map season"
            value={mapSeason}
            onChange={(event) => setMapSeason(event.target.value)}
          >
            <option value="">All seasons</option>
            {options?.seasons.map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
          <select
            aria-label="Map block"
            value={mapBlock}
            onChange={(event) => setMapBlock(event.target.value)}
          >
            <option value="">All blocks</option>
            {options?.blocks.map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
          <select
            aria-label="Map WQI class"
            value={mapClass}
            onChange={(event) => setMapClass(event.target.value)}
          >
            <option value="">All WQI classes</option>
            {options?.wqi_classes.map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </div>
        <MapPanel data={view} />
        <Legend counts={view?.metadata.display_class_counts ?? {}} />
        <div className="map-disclaimer">
          <AlertTriangle size={16} />
          {view?.metadata.coordinate_note} Calculated WQI is an index, not a
          potability certification; source labels remain separate.
        </div>
      </section>
    </div>
  );
}

function TemporalPage({
  temporal,
  analyte,
  setAnalyte,
  chart,
}: {
  temporal: Temporal | null;
  analyte: string;
  setAnalyte: (v: string) => void;
  chart: { year: number; mean: number | null }[];
}) {
  return (
    <div className="page-view">
      <Heading
        eyebrow="OBSERVED TIME SERIES"
        title="Temporal analysis"
        description="Annual and seasonal summaries calculated from actual observed measurements."
      />
      {!temporal ? (
        <Loading />
      ) : (
        <>
          <div className="time-summary">
            <span>
              <b>
                {temporal.time_coverage.first_year}–
                {temporal.time_coverage.last_year}
              </b>{" "}
              coverage
            </span>
            <span>
              <b>{temporal.time_coverage.years.length}</b> years
            </span>
            <span>
              <b>{temporal.time_coverage.seasons.length}</b> seasons
            </span>
            <span>
              <b>
                {temporal.record_count_by_year
                  .reduce((s, r) => s + r.record_count, 0)
                  .toLocaleString()}
              </b>{" "}
              records
            </span>
          </div>
          <section className="panel chart-panel">
            <div className="panel-header">
              <div>
                <div className="panel-kicker">ANNUAL OBSERVED MEAN</div>
                <h2>{labels[analyte]} over time</h2>
              </div>
              <select
                aria-label="Parameter"
                value={analyte}
                onChange={(e) => setAnalyte(e.target.value)}
              >
                {Object.entries(labels).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v}
                  </option>
                ))}
              </select>
            </div>
            <div className="chart-area">
              <Suspense fallback={<Loading />}>
                <TemporalChart data={chart} />
              </Suspense>
            </div>
            <div className="chart-caption">
              {analyte === "wqi"
                ? "Calculated weighted-arithmetic WQI from observed chemistry. Counts vary by year; this is not a causal or significance analysis."
                : "Observed arithmetic mean by year. Sample counts vary; this is not a causal or significance analysis."}
            </div>
          </section>
          <section className="panel yearly-table-panel">
            <div className="panel-kicker">CALCULATED INDEX CLASSES</div>
            <h2>WQI class counts by year</h2>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Year</th>
                    {[
                      "Excellent",
                      "Good",
                      "Poor",
                      "Very Poor",
                      "Unsuitable for Drinking",
                    ].map((className) => (
                      <th key={className}>{className}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {temporal.time_coverage.years.map((year) => (
                    <tr key={year}>
                      <td>{year}</td>
                      {["Excellent", "Good", "Poor", "Very Poor", "Unsuitable for Drinking"].map((className) => {
                        const count = temporal.annual_wqi_class_distribution.find(
                          (item) => item.year === year && item.wqi_class === className,
                        )?.record_count ?? 0;
                        return <td key={className}>{count}</td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <section className="panel yearly-table-panel">
            <div className="panel-kicker">SEASONAL COMPARISON</div>
            <h2>Observed {labels[analyte]} by year and season</h2>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Year</th>
                    {temporal.time_coverage.seasons.map((s) => (
                      <th key={s}>{s}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {temporal.time_coverage.years.map((y) => (
                    <tr key={y}>
                      <td>{y}</td>
                      {temporal.time_coverage.seasons.map((s) => {
                        const r = temporal.year_season_parameter_summary.find(
                          (v) => v.year === y && v.season === s,
                        );
                        return (
                          <td key={s}>
                            {r
                              ? `${number(r.parameters[analyte]?.mean, 2)} · n=${r.parameters[analyte]?.count ?? 0}`
                              : "—"}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <div className="limitation-list">
            <b>Interpretation limits</b>
            {temporal.limitations.map((v) => (
              <span key={v}>· {v}</span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function PredictionPage({
  options,
}: {
  options: PredictionOptions | null;
}) {
  const inputFields = [
    { key: "nitrate_mg_l", label: "Nitrate", unit: "mg/L" },
    { key: "fluoride_mg_l", label: "Fluoride", unit: "mg/L" },
    { key: "chloride_mg_l", label: "Chloride", unit: "mg/L" },
    { key: "tds_mg_l", label: "TDS", unit: "mg/L" },
    { key: "ph", label: "pH", unit: "pH units" },
    { key: "iron_mg_l", label: "Iron", unit: "mg/L" },
    { key: "alkalinity_mg_l", label: "Alkalinity", unit: "mg/L" },
  ] as const;
  const emptyValues = Object.fromEntries(
    inputFields.map(({ key }) => [key, ""]),
  );
  const [values, setValues] = useState<Record<string, string>>(emptyValues);
  const [sampleId, setSampleId] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [pending, setPending] = useState(false);
  const [requestError, setRequestError] = useState("");

  function updateValue(key: string, value: string) {
    setValues((current) => ({ ...current, [key]: value }));
  }

  function selectSample(sampleId: string) {
    const sample = options?.sample_profiles.find(({ id }) => id === sampleId);
    if (!sample) {
      setSampleId("");
      return;
    }
    setSampleId(sample.id);
    setValues(
      Object.fromEntries(
        inputFields.map(({ key }) => [key, String(sample.values[key])]),
      ),
    );
    setResult(null);
    setRequestError("");
  }

  function clearInputs() {
    setValues(emptyValues);
    setSampleId("");
    setResult(null);
    setRequestError("");
  }

  async function submitPrediction(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const payload = Object.fromEntries(
      inputFields.map(({ key }) => [key, Number(values[key])]),
    ) as Record<string, number>;
    if (inputFields.some(({ key }) => values[key].trim() === "") ||
      Object.values(payload).some((value) => !Number.isFinite(value))) {
      setRequestError("Enter a valid finite number for each of the seven parameters.");
      return;
    }
    if (Object.entries(payload).some(([key, value]) =>
      key === "ph" ? value < 0 || value > 14 : value < 0,
    )) {
      setRequestError("Concentrations must be zero or greater and pH must be between 0 and 14.");
      return;
    }
    setPending(true);
    setRequestError("");
    setResult(null);
    try {
      setResult(await post<Prediction>("/prediction", payload));
    } catch (error) {
      setRequestError(
        error instanceof Error
          ? error.message
          : "Unable to generate this estimate. Check the entered values and try again.",
      );
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="page-view">
      <Heading
        title="WQI prediction"
      />
      {!options ? (
        <Loading />
      ) : (
        <div className="prediction-layout">
          <form className="panel prediction-form" onSubmit={submitPrediction}>
            <div className="panel-kicker">CHEMISTRY INPUTS</div>
            <h2>Enter a water sample</h2>
            <label>
              Sample values
              <select
                aria-label="Sample values"
                value={sampleId}
                onChange={(event) => selectSample(event.target.value)}
              >
                <option value="">Choose a profile…</option>
                {options.sample_profiles.map((sample) => (
                  <option key={sample.id} value={sample.id}>{sample.label}</option>
                ))}
              </select>
            </label>
            <div className="field-grid">
              {inputFields.map(({ key, label, unit }) => (
                <label key={key}>
                  {label} ({unit})
                  <input
                    type="number"
                    step="any"
                    min="0"
                    max={key === "ph" ? "14" : undefined}
                    value={values[key]}
                    onChange={(event) => updateValue(key, event.target.value)}
                    placeholder={`Enter ${unit}`}
                    required
                  />
                </label>
              ))}
            </div>
            <div className="prediction-form-actions">
              <button
                className="primary-button prediction-submit"
                type="submit"
                disabled={pending}
              >
                {pending ? "Analyzing…" : "Predict WQI"}
              </button>
              <button className="clear-button" type="button" onClick={clearInputs}>
                Clear inputs
              </button>
            </div>
            {requestError && <p className="form-error">{requestError}</p>}
          </form>
          <section className="panel prediction-result prediction-result-panel">
            <div className="prediction-result-heading">
              <div>
                <h2>Sample assessment</h2>
              </div>
              {result && (
                <button className="clear-button" onClick={clearInputs}>
                  Clear result
                </button>
              )}
            </div>
            {result && (
              <div className="prediction-result-content">
                <div className="score-summary">
                  <div className="score-card">
                    <span>Model-estimated WQI</span>
                    <strong>{number(result.predicted_wqi, 2)}</strong>
                    <b
                      className="prediction-class"
                      style={{ "--class-color": color(result.predicted_class) } as React.CSSProperties}
                    >
                      {result.predicted_class}
                    </b>
                  </div>
                  <div className="score-card calculated-score">
                    <span>Formula-calculated WQI</span>
                    <strong>{number(result.calculated_wqi, 2)}</strong>
                    <b
                      className="prediction-class"
                      style={{ "--class-color": color(result.calculated_class) } as React.CSSProperties}
                    >
                      {result.calculated_class}
                    </b>
                  </div>
                </div>
                <div className="parameter-assessment">
                  <div className="panel-kicker">INDIVIDUAL BIS GUIDELINE CHECKS</div>
                  <div className="parameter-assessment-list">
                    {result.parameter_assessments.map((item) => (
                      <article className={`assessment-row assessment-${item.tone}`} key={item.field}>
                        <div className="assessment-name">
                          <strong>{item.label}</strong>
                          <span>{number(item.value, 3)} {item.unit}</span>
                        </div>
                        <div className="assessment-status">
                          <b>{item.status}</b>
                          <span>Reference: {item.reference}</span>
                        </div>
                      </article>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

function Reports({
  summary,
  temporal,
  download,
}: {
  summary: Summary;
  temporal: Temporal | null;
  download: () => void;
}) {
  return (
    <div className="page-view">
      <Heading
        eyebrow="RESEARCH OUTPUTS"
        title="Reports & provenance"
        description="Download a machine-readable snapshot of supportable observed-data results."
        action={
          <button className="primary-button" onClick={download}>
            <ArrowDownToLine size={16} /> Download report
          </button>
        }
      />
      <div className="report-grid">
        <section className="panel report-section">
          <div className="panel-kicker">DATASET SUMMARY</div>
          <h2>Included observations</h2>
          <dl>
            <div>
              <dt>Complete records</dt>
              <dd>{summary.record_count.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Source blocks</dt>
              <dd>{summary.block_count}</dd>
            </div>
            <div>
              <dt>Villages represented</dt>
              <dd>{summary.village_count}</dd>
            </div>
            <div>
              <dt>Mean calculated WQI</dt>
              <dd>{number(summary.wqi_summary.mean_wqi, 2)}</dd>
            </div>
            <div>
              <dt>Temporal coverage</dt>
              <dd>{summary.year_range?.join("–") ?? "Unavailable"}</dd>
            </div>
            <div>
              <dt>Incomplete source rows</dt>
              <dd>
                {summary.preprocessing.quarantined_malformed_records ?? "—"}
              </dd>
            </div>
            <div>
              <dt>Exact duplicates removed</dt>
              <dd>{summary.preprocessing.removed_exact_duplicates ?? "—"}</dd>
            </div>
          </dl>
        </section>
        <section className="panel report-section">
          <div className="panel-kicker">ANALYSIS STATUS</div>
          <h2>Available vs. pending</h2>
          <dl>
            <div>
              <dt>Observed summaries</dt>
              <dd className="status-ready">Available</dd>
            </div>
            <div>
              <dt>Spatial point sample</dt>
              <dd className="status-ready">Optimized</dd>
            </div>
            <div>
              <dt>Temporal chemistry</dt>
              <dd className="status-ready">
                {temporal ? "Available" : "Load analysis"}
              </dd>
            </div>
            <div>
              <dt>WQI and classes</dt>
              <dd className="status-ready">Calculated · 6 parameters</dd>
            </div>
            <div>
              <dt>RF/XGBoost and SHAP</dt>
              <dd className="status-ready">Available · weak R²</dd>
            </div>
          </dl>
        </section>
      </div>
      <section className="panel report-limitations">
        <div className="panel-kicker">LIMITATIONS</div>
        <h2>Read before interpretation</h2>
        <ul>
          <li>
            Source files have no headers; field names are inferred by position.
          </li>
          <li>
            Coordinates are not confirmed as Cuddalore locations and are not
            checked against an official boundary.
          </li>
          <li>
            Existing quality/risk labels are undocumented source categories, not
            calculated WQI classes.
          </li>
          <li>
            WQI uses BIS IS 10500:2012 acceptable limits and a published weighted-arithmetic method; alkalinity is excluded because its reporting basis is unspecified.
          </li>
          <li>
            Grouped holdout R² is negative for both models; the predictors currently provide weak estimates.
          </li>
        </ul>
      </section>
    </div>
  );
}
