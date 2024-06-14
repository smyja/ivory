"use client"
import { useSearchParams,useRouter } from 'next/navigation';
import { useConnectNotionMutation } from '@/redux/features/authApiSlice';
import { useEffect } from 'react';


const Callback = () => {
  const [connectNotion, { isLoading }] = useConnectNotionMutation();
  const searchParams = useSearchParams();
  const router = useRouter();

  useEffect(() => {
    const handleConnect = async (authCode:string) => {
        console.log("Handling connection with auth code:", authCode);
      try {
        await connectNotion(authCode).unwrap();
        // Redirect to dashboard or display success message
        // router.push('/dashboard'); // Change the route accordingly
        console.log("success")
      } catch (err) {
        console.error(err);
        // Display error message to the user
        // You can use state or a notification system for this
      }
    };

    if (searchParams) {
      const authCode = searchParams.get('code');

      if (authCode) {
        handleConnect(authCode);
      }
    }
  }, [searchParams, connectNotion, router]);

  return (
    <div>
      {isLoading ? (
        <p>Connecting...</p>
      ) : (
        <p>Connected! and Syncing..</p>
      )}
    </div>
  );
};

export default Callback;
