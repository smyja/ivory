'use client';

import { useState } from 'react';
import { useDisclosure } from '@mantine/hooks';
import { AppShell, Burger, Group, Skeleton, Box, ActionIcon } from '@mantine/core';
import { NavbarSimpleColored } from '@/components/Navbar/NavbarSimpleColored';
import { DownloadProvider } from '@/components/DownloadNotifications/DownloadContext';
import { DownloadNotificationsCenter } from '@/components/DownloadNotifications/DownloadNotificationsCenter';

export default function DashboardLayout({
  children, // will be a page or nested layout
}: {
  children: React.ReactNode;
}) {
  const [opened, { toggle }] = useDisclosure();
  const [desktopSidebarCollapsed, setDesktopSidebarCollapsed] = useState(false);

  return (
    <DownloadProvider>
      <AppShell
        header={{ height: 60 }}
        navbar={{
          width: 220,
          breakpoint: 'sm',
          collapsed: { mobile: !opened, desktop: desktopSidebarCollapsed },
        }}
        padding="md"
        layout="alt"
        style={{ backgroundColor: '#fafafa' }}
      >
        <AppShell.Header style={{ backgroundColor: 'white' }}>
          <Group h="100%" px="md" justify="space-between">
            <Group>
              <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
              <ActionIcon
                variant="subtle"
                color="gray"
                aria-label="Toggle sidebar"
                onClick={() => setDesktopSidebarCollapsed((prev) => !prev)}
                ml="xs"
              >
                <svg
                  aria-hidden="true"
                  width="16"
                  height="16"
                  viewBox="0 0 16 16"
                  fill="currentColor"
                  role="img"
                  focusable="false"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    fillRule="evenodd"
                    clipRule="evenodd"
                    d="M4.25 2C2.45508 2 1 3.45508 1 5.25V10.7499C1 12.5449 2.45508 13.9999 4.25 13.9999H11.75C13.5449 13.9999 15 12.5449 15 10.7499V5.25C15 3.45508 13.5449 2 11.75 2H4.25ZM2.5 10.4999C2.5 11.6045 3.39543 12.4999 4.5 12.4999H11.75C12.7165 12.4999 13.5 11.7164 13.5 10.7499V5.25C13.5 4.28351 12.7165 3.5 11.75 3.5H4.5C3.39543 3.5 2.5 4.39543 2.5 5.5V10.4999Z"
                  ></path>
                  <rect x="9" y="3" width="1.5" height="10"></rect>
                </svg>
              </ActionIcon>
              IVORY
            </Group>
            <Box mr="xl">
              <DownloadNotificationsCenter />
            </Box>
          </Group>
        </AppShell.Header>
        <AppShell.Navbar p={0} style={{ backgroundColor: 'midnightblue' }}>
          <NavbarSimpleColored />
        </AppShell.Navbar>

        <AppShell.Main>{children}</AppShell.Main>
      </AppShell>
    </DownloadProvider>
  );
}
