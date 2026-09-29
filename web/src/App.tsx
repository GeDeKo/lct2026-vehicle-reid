import { useEffect, useState } from "react";
import { enroll, fetchHealth, identify } from "./api";
import { BrandMark } from "./components/BrandMark";
import { CapturedFrame } from "./components/CapturedFrame";
import { Dropzone } from "./components/Dropzone";
import { MatchTile } from "./components/MatchTile";
import { ProfilePanel } from "./components/ProfilePanel";
import { formatNow } from "./format";
import type { IdentifyResponse } from "./types";

type Health = { ok: boolean; indexSize: number };

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [capturedAt, setCapturedAt] = useState<string>("");
  const [result, setResult] = useState<IdentifyResponse | null>(null);
  const [identifying, setIdentifying] = useState(false);
  const [enrolling, setEnrolling] = useState(false);
  const [cameraId, setCameraId] = useState("");
  const [enrollNote, setEnrollNote] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function refreshHealth() {
    try {
      const h = await fetchHealth();
      setHealth({ ok: true, indexSize: h.index_size });
    } catch {
      setHealth({ ok: false, indexSize: 0 });
    }
  }

  useEffect(() => {
    refreshHealth();
  }, []);

  function handleFile(next: File) {
    setFile(next);
    setPreviewUrl(URL.createObjectURL(next));
    setCapturedAt(formatNow());
    setResult(null);
    setError(null);
    setEnrollNote("");
  }

  async function handleIdentify() {
    if (!file) return;
    setError(null);
    setIdentifying(true);
    setResult(null);
    try {
      setResult(await identify(file));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIdentifying(false);
    }
  }

  async function handleEnroll() {
    if (!file) return;
    setError(null);
    setEnrolling(true);
    setEnrollNote("");
    try {
      const { record_id } = await enroll(file, cameraId.trim() || "unknown");
      setEnrollNote(`Добавлено в базу, запись №${record_id}`);
      refreshHealth();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setEnrolling(false);
    }
  }

  return (
    <>
      <header className="shell-header">
        <div className="brand">
          <BrandMark />
          <span className="brand-name">Re-ID console</span>
        </div>
        <div className="status">
          <span className={`status-dot${health ? (health.ok ? " ok" : " down") : ""}`} />
          {health === null && "проверяю API…"}
          {health?.ok && `API: ${health.indexSize} записей в базе`}
          {health && !health.ok && "API недоступен — проверьте проброс порта"}
        </div>
      </header>

      <div className="console">
        <section className="pane pane-capture">
          <Dropzone onFile={handleFile} />

          {previewUrl && (
            <>
              <CapturedFrame src={previewUrl} tag={`КАДР · ${capturedAt}`} scanning={identifying} />

              <div className="actions">
                <div className="actions-row">
                  <button className="btn btn-primary" onClick={handleIdentify} disabled={identifying}>
                    {identifying ? "Ищу…" : "Найти похожие"}
                  </button>
                </div>
                <div className="actions-row">
                  <input
                    className="camera-input"
                    placeholder="camera_id"
                    value={cameraId}
                    onChange={(e) => setCameraId(e.target.value)}
                  />
                  <button className="btn btn-secondary" onClick={handleEnroll} disabled={enrolling}>
                    {enrolling ? "Добавляю…" : "Добавить в базу"}
                  </button>
                </div>
                {enrollNote && <div className="enroll-note">{enrollNote}</div>}
              </div>

              {error && <div className="error-banner">Не вышло: {error}</div>}

              {result && <ProfilePanel profile={result.profile} />}
            </>
          )}
        </section>

        <section className="pane pane-results">
          <h2>Совпадения</h2>

          {identifying && (
            <div className="matches-grid">
              {Array.from({ length: 6 }).map((_, i) => (
                <div className="skeleton-tile" key={i} />
              ))}
            </div>
          )}

          {!identifying && result && result.matches.length === 0 && (
            <div className="results-empty">Совпадений не найдено — база пуста или машина уникальна.</div>
          )}

          {!identifying && result && result.matches.length > 0 && (
            <div className="matches-grid">
              {result.matches.map((m, i) => (
                <MatchTile match={m} index={i} key={`${m.camera_id}-${m.timestamp}-${i}`} />
              ))}
            </div>
          )}

          {!identifying && !result && (
            <div className="results-empty">Загрузите фото слева, чтобы найти похожие машины.</div>
          )}
        </section>
      </div>
    </>
  );
}
