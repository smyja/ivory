export async function generateStaticParams() {
  // For now, we'll pre-render a few example dataset IDs
  return [{ id: '1' }, { id: '2' }, { id: '3' }];
}

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
