import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = request.nextUrl;
    const dataset_name = searchParams.get('dataset_name');
    const token = searchParams.get('token');

    if (!dataset_name) {
      return NextResponse.json(
        { error: 'Missing required parameter: dataset_name' },
        { status: 400 }
      );
    }

    // Forward the request to the FastAPI backend
    const apiUrl = new URL(`${process.env.NEXT_PUBLIC_API_URL}/datasets/huggingface/configs`);
    apiUrl.searchParams.append('dataset_name', dataset_name);
    if (token) {
      apiUrl.searchParams.append('token', token);
    }

    const response = await fetch(apiUrl.toString(), {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch dataset configs' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching Hugging Face dataset configs:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
