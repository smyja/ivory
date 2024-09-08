'use client'
import { useDisclosure } from '@mantine/hooks';
import { AppShell, Burger, Group, Skeleton } from '@mantine/core';
import { NavbarSimpleColored } from '@/components/Navbar/NavbarSimpleColored';

export default function DashboardLayout({
  children, // will be a page or nested layout
}: {
  children: React.ReactNode
}) {
  const [opened, { toggle }] = useDisclosure();

  return (

      <AppShell
        header={{ height: 60 }}
        navbar={{ width: 220, breakpoint: 'sm', collapsed: { mobile: !opened } }}
        padding="md"
        layout='alt'
        style={{backgroundColor: "#fafafa"}}
      >
        <AppShell.Header
         style={{backgroundColor: "white"}}
        >
          <Group h="100%" px="md">
            <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
            IVORY
          </Group>
        </AppShell.Header>
        <AppShell.Navbar p="sm"
        style={{backgroundColor: "white"}}
        >
          <NavbarSimpleColored />
          
        </AppShell.Navbar>
   
        <AppShell.Main>
          {children}
        </AppShell.Main>
      
      </AppShell>
  
  )
}