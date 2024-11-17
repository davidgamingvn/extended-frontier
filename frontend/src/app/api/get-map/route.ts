// app/api/get-map/route.ts
import { NextResponse } from 'next/server'

export async function GET() {
  try {
    // Gage?
    const imageUrl = '/updated_image.jpg'
    return NextResponse.json({ imageUrl })
  } catch (error) {
    return NextResponse.json({ error: 'Failed to get updated map' }, { status: 500 })
  }
}
