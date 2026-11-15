"use client";

import { useEffect, useRef, useState } from "react";
import { loadYouTubeApi, playerErrorMessage } from "@/lib/youtube";

export interface UseYouTubePlayerOptions {
  videoId: string;
  /** Seconds to start/seek to. */
  start: number;
  /** Increment to replay the same timestamp when it is clicked again. */
  seekKey?: number;
}

export interface UseYouTubePlayerResult {
  containerRef: React.RefObject<HTMLDivElement | null>;
  ready: boolean;
  error: string | null;
}

export function useYouTubePlayer({
  videoId,
  start,
  seekKey = 0,
}: UseYouTubePlayerOptions): UseYouTubePlayerResult {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const playerRef = useRef<YTPlayer | null>(null);
  const currentVideoRef = useRef<string | null>(null);
  const latestRef = useRef({ videoId, start });
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    latestRef.current = { videoId, start };
  }, [videoId, start]);

  // Create the player once; video changes are handled imperatively below so
  // same-video timestamp clicks only seek instead of reloading the iframe.
  useEffect(() => {
    let disposed = false;
    const host = containerRef.current;
    if (!host) return;

    // The IFrame API replaces the element it is given with an <iframe>. Give it
    // an imperatively created child so React still owns the container node and
    // never tries to remove a node the player already replaced.
    const mount = document.createElement("div");
    mount.className = "h-full w-full";
    host.appendChild(mount);

    loadYouTubeApi()
      .then((YT) => {
        if (disposed) return;

        const player = new YT.Player(mount, {
          videoId: latestRef.current.videoId,
          host: "https://www.youtube-nocookie.com",
          width: "100%",
          height: "100%",
          playerVars: {
            rel: 0,
            modestbranding: 1,
            playsinline: 1,
            start: latestRef.current.start,
            origin: window.location.origin,
          },
          events: {
            onReady: () => {
              if (!disposed) setReady(true);
            },
            onError: (event) => {
              if (!disposed) setError(playerErrorMessage(event.data ?? 2));
            },
          },
        });

        playerRef.current = player;
        currentVideoRef.current = latestRef.current.videoId;
      })
      .catch(() => {
        if (!disposed) {
          setError(
            "Could not load the YouTube player. Check your connection and try again.",
          );
        }
      });

    return () => {
      disposed = true;
      const player = playerRef.current;
      playerRef.current = null;
      currentVideoRef.current = null;
      if (player) {
        try {
          player.destroy();
        } catch {
          // The iframe may already be gone (React strict-mode remount).
        }
      }
      mount.remove();
    };
  }, []);

  useEffect(() => {
    latestRef.current = { videoId, start };
    if (!ready) return;

    const player = playerRef.current;
    if (!player) return;

    if (currentVideoRef.current !== videoId) {
      currentVideoRef.current = videoId;
      setError(null);
      player.loadVideoById({ videoId, startSeconds: start });
      return;
    }

    player.seekTo(start, true);
    player.playVideo();
  }, [videoId, start, seekKey, ready]);

  return { containerRef, ready, error };
}
