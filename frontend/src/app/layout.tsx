import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/AppShell";

export const metadata: Metadata = {
  title: "Bio-Digital Twin | Drug Discovery Platform",
  description:
    "End-to-end drug discovery platform — synthetic patients, GNN, disease dynamics, causal AI, pharmacogenomics, PK/PD, clinical trials, and explainable AI. 16 phases.",
  keywords: ["digital twin", "drug discovery", "pharmacogenomics", "clinical trials", "AI"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
