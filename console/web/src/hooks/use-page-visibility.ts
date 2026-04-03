"use client";

import { useEffect, useState } from "react";

function readVisibility(): boolean {
  if (typeof document === "undefined") {
    return true;
  }

  return document.visibilityState !== "hidden";
}

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
