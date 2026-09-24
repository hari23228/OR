import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = { title: "Cost-Aware LLM Routing", description: "Operations Research routing optimization" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
