/**
 * Timestamp formatting.
 *
 * Under an hour: `mm:ss` (zero-padded, matching the backend's `start_label`).
 * At or above an hour: `h:mm:ss`.
 */
export function formatTimestamp(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) return "00:00";

  const whole = Math.floor(totalSeconds);
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const seconds = whole % 60;

  const ss = String(seconds).padStart(2, "0");
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${ss}`;
  }
  return `${String(minutes).padStart(2, "0")}:${ss}`;
}
