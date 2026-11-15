export {};

declare global {
  interface YTPlayer {
    loadVideoById(options: { videoId: string; startSeconds?: number }): void;
    seekTo(seconds: number, allowSeekAhead: boolean): void;
    playVideo(): void;
    destroy(): void;
  }

  interface YTPlayerEvent {
    target: YTPlayer;
    data?: number;
  }

  interface YTPlayerOptions {
    videoId: string;
    host?: string;
    width?: string | number;
    height?: string | number;
    playerVars?: Record<string, string | number>;
    events?: {
      onReady?: (event: YTPlayerEvent) => void;
      onError?: (event: YTPlayerEvent) => void;
      onStateChange?: (event: YTPlayerEvent) => void;
    };
  }

  interface YTNamespace {
    Player: new (element: HTMLElement, options: YTPlayerOptions) => YTPlayer;
  }

  interface Window {
    YT?: YTNamespace;
    onYouTubeIframeAPIReady?: () => void;
  }
}
