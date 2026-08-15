import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "47.245M Qwen3 · Pretrain → SFT → RL Comparison",
  description: "Live reward and pass@k curves for two pretrain-to-SFT checkpoints and two RL learning rates.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
