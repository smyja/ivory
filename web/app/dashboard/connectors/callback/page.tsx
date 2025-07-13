'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import { useEffect } from 'react';
import { useConnectNotionMutation } from '@/redux/features/authApiSlice';

// For static export, we'll create a placeholder page
const Callback = () => {
  const searchParams = useSearchParams();
  const router = useRouter();

  useEffect(() => {
    const authCode = searchParams?.get('code');
    if (authCode) {
      // Store the auth code in localStorage or handle it client-side
      localStorage.setItem('notion_auth_code', authCode);
      // Redirect to a page that can handle the connection
      router.push('/dashboard/connectors?status=connecting');
    } else {
      router.push('/dashboard/connectors?status=error');
    }
  }, [searchParams, router]);

  return (
    <div className="flex items-center justify-center min-h-screen">
      <p className="text-lg">Connecting to Notion...</p>
    </div>
  );
};

export default Callback;
