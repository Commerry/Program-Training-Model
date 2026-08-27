/** Formatting helpers shared by the views. */

export const formatDateTime = (value) => {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

export const formatDate = (value) => {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  })
}

/** Elapsed time between two ISO timestamps, as "2h 15m". */
export const formatDuration = (from, to) => {
  if (!from) return '—'
  const start = new Date(from).getTime()
  const end = to ? new Date(to).getTime() : Date.now()
  if (Number.isNaN(start) || Number.isNaN(end) || end < start) return '—'
  const seconds = Math.round((end - start) / 1000)
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  if (hours) return `${hours}h ${minutes}m`
  if (minutes) return `${minutes}m ${seconds % 60}s`
  return `${seconds}s`
}

/** A 0-1 metric as a percentage string. Returns '—' when it was never measured. */
export const formatMetric = (value, digits = 1) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return `${(Number(value) * 100).toFixed(digits)}%`
}

/** A 0-1 metric as a bare number 0-100, for progress-bar widths. */
export const metricPercent = (value) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0
  return Math.max(0, Math.min(100, Number(value) * 100))
}

export const formatBytes = (bytes) => {
  if (!bytes && bytes !== 0) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = Number(bytes)
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`
}

export const formatNumber = (value) =>
  Number(value || 0).toLocaleString()

/** Maps a run status to the CSS badge modifier the stylesheet defines. */
export const statusVariant = (status) => ({
  completed: 'success',
  running: 'warning',
  preparing: 'warning',
  stopping: 'warning',
  stopped: 'neutral',
  idle: 'neutral',
  failed: 'danger'
}[status] || 'neutral')
