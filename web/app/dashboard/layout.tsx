'use client'
import { useDisclosure } from '@mantine/hooks';
import { AppShell, Burger, Group, Skeleton } from '@mantine/core';
import { NavbarSimpleColored } from '@/components/Navbar/NavbarSimpleColored';
import { RequireAuth } from '@/components/utils';

export default function DashboardLayout({
  children, // will be a page or nested layout
}: {
  children: React.ReactNode
}) {
  const [opened, { toggle }] = useDisclosure();

  return (
    <RequireAuth>
      <AppShell
        header={{ height: 60 }}
        navbar={{ width: 300, breakpoint: 'sm', collapsed: { mobile: !opened } }}
        padding="md"
        layout='alt'
      >
        <AppShell.Header>
          <Group h="100%" px="md">
            <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
            Railsearch
          </Group>
        </AppShell.Header>
        <AppShell.Navbar p="sm">
          <NavbarSimpleColored />
          
        </AppShell.Navbar>
   
        <AppShell.Main>
          {children}
        </AppShell.Main>
      
      </AppShell>
      </RequireAuth>
  )
}