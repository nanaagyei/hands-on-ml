/**
 * API prefix: in Vite dev, `/api` is proxied to Flask with `/api` stripped.
 * Production build served from Flask uses same origin; paths are root-relative (`/health`).
 * Override with VITE_API_BASE if needed.
 */
export function apiBase(): string {
  const explicit = import.meta.env.VITE_API_BASE
  if (explicit !== undefined && explicit !== '') return explicit
  return import.meta.env.DEV ? '/api' : ''
}

export function apiUrl(path: string): string {
  const base = apiBase()
  const p = path.startsWith('/') ? path : `/${path}`
  return `${base}${p}`
}
