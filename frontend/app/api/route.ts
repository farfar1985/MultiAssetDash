import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    message: "QDT Nexus BFF API",
    version: "0.1.0",
    status: "placeholder",
  });
}
