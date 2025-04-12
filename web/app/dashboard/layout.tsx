'use client'
import { useDisclosure } from '@mantine/hooks';
import { AppShell, Burger, Group, Skeleton, Box } from '@mantine/core';
import { NavbarSimpleColored } from '@/components/Navbar/NavbarSimpleColored';
import { DownloadProvider } from '@/components/DownloadNotifications/DownloadContext';
import { DownloadNotificationsCenter } from '@/components/DownloadNotifications/DownloadNotificationsCenter';

export default function DashboardLayout({
  children, // will be a page or nested layout
}: {
  children: React.ReactNode
}) {
  const [opened, { toggle }] = useDisclosure();

  return (
    <DownloadProvider>
      <AppShell
        header={{ height: 60 }}
        navbar={{ width: 220, breakpoint: 'sm', collapsed: { mobile: !opened } }}
        padding="md"
        layout='alt'
        style={{ backgroundColor: "#fafafa" }}
      >
        <AppShell.Header
          style={{ backgroundColor: "white" }}
        >
          <Group h="100%" px="md" justify="space-between">
            <Group>
              <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
              IVORY
            </Group>
            <Box mr="xl">
              <DownloadNotificationsCenter />
            </Box>
          </Group>
        </AppShell.Header>
        <AppShell.Navbar p="sm"
          style={{ backgroundColor: "black" }}
        >
          <NavbarSimpleColored />

        </AppShell.Navbar>

        <AppShell.Main>
          {children}
        </AppShell.Main>

      </AppShell>
    </DownloadProvider>
  )
}