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

      <AppShell
        header={{ height: 60 }}
        navbar={{ width: 300, breakpoint: 'sm', collapsed: { mobile: !opened } }}
        padding="md"
        layout='alt'
        style={{backgroundColor: "#f5f5f5"}}
      >
        <AppShell.Header
         style={{backgroundColor: "#f5f5f5"}}
        >
          <Group h="100%" px="md">
            <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
            IVORY
          </Group>
        </AppShell.Header>
        <AppShell.Navbar p="sm"
        style={{backgroundColor: "#f5f5f5"}}
        >
          <NavbarSimpleColored />
          
        </AppShell.Navbar>
   
        <AppShell.Main>
          {children}
        </AppShell.Main>
      
      </AppShell>
  
  )
}