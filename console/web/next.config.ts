import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["127.0.0.1", "10.0.18.131", "localhost"],
  output: "standalone",
};

export default nextConfig;
