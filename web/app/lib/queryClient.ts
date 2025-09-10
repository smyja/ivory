import { API_BASE_URL, API_ENDPOINTS } from '../config/api';

export type WhereCond = { column: string; op: string; value: any };
export type OrderBy = { column: string; direction?: 'asc' | 'desc' };

export async function previewDataset(dataset: string) {
  const res = await fetch(`${API_BASE_URL}${API_ENDPOINTS.query.preview(dataset)}`);
  if (!res.ok) throw new Error(`Preview failed: ${res.status}`);
  return res.json();
}

export async function runQuery(params: {
  dataset: string;
  select?: string[];
  where?: WhereCond[];
  order_by?: OrderBy;
  limit?: number;
  offset?: number;
}) {
  const res = await fetch(`${API_BASE_URL}${API_ENDPOINTS.query.run}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    let msg = `Query failed: ${res.status}`;
    try {
      const j = await res.json();
      msg = j?.detail || msg;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

export async function upsertLabelByRowId(dataset: string, label: string, row_id: string, value: string, text?: string) {
  const res = await fetch(`${API_BASE_URL}${API_ENDPOINTS.query.labelUpsertRow}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset, label, row_id, value, text }),
  });
  if (!res.ok) throw new Error(`Label upsert failed: ${res.status}`);
  return res.json();
}

