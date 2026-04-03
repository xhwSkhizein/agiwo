"use client";

import { useEffect, useState } from "react";

/**
 * Determine whether the page should be considered visible in the current environment.
 *
 * If `document` is unavailable (for example during server-side rendering) this function treats the page as visible.
 *
 * @returns `true` if `document` is unavailable or `document.visibilityState` is not `"hidden"`, `false` otherwise.
 */
function readVisibility(): boolean {
  if (typeof document === "undefined") {
    return true;
  }

  return document.visibilityState !== "hidden";
}

/**
 * Tracks whether the page is considered visible and updates when visibility or focus changes.
 *
 * @returns `true` if the page is currently visible, `false` otherwise. When `document` is unavailable (for example during server-side rendering), returns `true`.
 */
export function usePageVisibility(): boolean {
  const [isVisible, setIsVisible] = useState(readVisibility);

  useEffect(() => {
    const handleVisibilityChange = () => {
      setIsVisible(readVisibility());
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    window.addEventListener("focus", handleVisibilityChange);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      window.removeEventListener("focus", handleVisibilityChange);
    };
  }, []);

  return isVisible;
}

export default usePageVisibility;
