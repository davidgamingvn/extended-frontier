// app/api/save-map/route.ts
import { NextResponse } from 'next/server'

export async function POST(request: Request) {
  try {
    const data = await request.json()
    // Gage?

    return NextResponse.json({ message: 'Map saved successfully' })
  } catch (error) {
    return NextResponse.json({ error: 'Failed to save map' }, { status: 500 })
  }
}
