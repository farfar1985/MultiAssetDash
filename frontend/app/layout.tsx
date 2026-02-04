import type { Metadata } from "next";
import localFont from "next/font/local";
import { QueryProvider } from "@/providers/QueryProvider";
import { EnsembleProvider } from "@/contexts/EnsembleContext";
import { CommandPaletteWrapper } from "@/components/ai";
import "./globals.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "QDT Nexus | Quantum Decision Theory Dashboard",
  description: "Multi-persona commodity analytics powered by Quantum Decision Theory",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-neutral-950 text-neutral-100`}
      >
        <QueryProvider>
          <EnsembleProvider>
            <CommandPaletteWrapper>{children}</CommandPaletteWrapper>
          </EnsembleProvider>
        </QueryProvider>
      </body>
    </html>
  );
}
