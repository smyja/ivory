import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const requestData = await request.json();

    // Forward the request to the FastAPI backend
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/datasets`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestData),
    });

    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.detail || 'Failed to create dataset' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error creating dataset:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
