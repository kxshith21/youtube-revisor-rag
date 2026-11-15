/** YouTube helpers: source-URL parsing, embed loader, and player error copy. */

const YOUTUBE_ID = /^[A-Za-z0-9_-]{11}$/;

export interface SourceVideo {
  videoId: string;
  start: number;
}

/**
 * Backend sources use `https://youtu.be/<id>?t=<seconds>`.
 * `start_label` is for display only; the `t` parameter is authoritative.
 */
export function parseSourceVideo(url: string): SourceVideo | null {
  if (!url) return null;

  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return null;
  }

  let videoId = "";
  const host = parsed.hostname.toLowerCase();

  if (host === "youtu.be" || host.endsWith(".youtu.be")) {
    videoId = parsed.pathname.split("/").filter(Boolean)[0] ?? "";
  } else if (host.endsWith("youtube.com") || host.endsWith("youtube-nocookie.com")) {
    videoId = parsed.searchParams.get("v") ?? "";
  }

  if (!YOUTUBE_ID.test(videoId)) return null;

  const rawStart = parsed.searchParams.get("t") ?? "";
  const parsedStart = Number.parseInt(rawStart, 10);
  const start =
    Number.isFinite(parsedStart) && parsedStart > 0 ? parsedStart : 0;

  return { videoId, start };
}

/** Public watch URL used by "Open on YouTube" links. */
export function watchUrl(videoId: string, start = 0): string {
  const suffix = start > 0 ? `&t=${start}s` : "";
  return `https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}${suffix}`;
}

const PLAYER_ERRORS: Record<number, string> = {
  2: "This video ID is invalid.",
  5: "This video can't be played in an embedded player.",
  100: "This video was removed or is private.",
  101: "The video owner disabled embedding.",
  150: "The video owner disabled embedding.",
};

export function playerErrorMessage(code: number): string {
  return PLAYER_ERRORS[code] ?? "This video can't be played here.";
}

let apiPromise: Promise<YTNamespace> | null = null;

/** Load the IFrame API once, reusing a script that is already on the page. */
export function loadYouTubeApi(): Promise<YTNamespace> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("The YouTube player needs a browser."));
  }
  if (window.YT?.Player) return Promise.resolve(window.YT);
  if (apiPromise) return apiPromise;

  apiPromise = new Promise<YTNamespace>((resolve, reject) => {
    let settled = false;
    let poll = 0;

    const finish = (namespace?: YTNamespace) => {
      if (settled) return;
      settled = true;
      if (poll) window.clearInterval(poll);
      if (namespace) resolve(namespace);
      else reject(new Error("Could not load the YouTube player."));
    };

    const previous = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      previous?.();
      if (window.YT?.Player) finish(window.YT);
    };

    const existing = document.querySelector<HTMLScriptElement>(
      'script[src="https://www.youtube.com/iframe_api"]',
    );

    if (!existing) {
      const script = document.createElement("script");
      script.src = "https://www.youtube.com/iframe_api";
      script.async = true;
      script.onerror = () => finish();
      document.head.appendChild(script);
    }

    // Cover the case where the script is already present (or was injected by
    // another component): poll until the namespace appears.
    let attempts = 0;
    poll = window.setInterval(() => {
      attempts += 1;
      if (window.YT?.Player) finish(window.YT);
      else if (attempts > 100) finish();
    }, 100);
  });

  apiPromise.catch(() => {
    apiPromise = null;
  });
  return apiPromise;
}
