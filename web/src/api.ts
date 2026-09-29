import type { HealthResponse, IdentifyResponse } from "./types";

// Сайт статический, крутится на ноуте; инференс — на A100. По умолчанию
// ждём проброшенный порт (см. README), можно переопределить через
// ?api=http://host:port или VITE_API_BASE при сборке.
export const API_BASE =
  new URLSearchParams(location.search).get("api") ||
  import.meta.env.VITE_API_BASE ||
  "http://localhost:8800";

class ApiError extends Error {}

async function parseError(res: Response): Promise<never> {
  const body = await res.json().catch(() => null);
  throw new ApiError(body?.detail || `Ошибка ${res.status}`);
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) return parseError(res);
  return res.json();
}

export async function identify(file: File): Promise<IdentifyResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/identify`, { method: "POST", body: form });
  if (!res.ok) return parseError(res);
  return res.json();
}

export async function enroll(file: File, cameraId: string): Promise<{ record_id: number }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/enroll?camera_id=${encodeURIComponent(cameraId)}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) return parseError(res);
  return res.json();
}

export function assetUrl(path: string): string {
  return `${API_BASE}/${path}`;
}
