'use client';

import { useState } from 'react';
import {
    Indicator,
    Menu,
    ActionIcon,
    Text,
    Progress,
    Group,
    Paper,
    Badge,
    Divider,
    Box,
    Title,
    Button
} from '@mantine/core';
import { IconDownload, IconCheck, IconX, IconTrash } from '@tabler/icons-react';
import { useRouter } from 'next/navigation';
import { useDownloads } from './DownloadContext';

export function DownloadNotificationsCenter() {
    const [opened, setOpened] = useState(false);
    const { downloads, clearCompleted } = useDownloads();
    const router = useRouter();

    const handleViewDataset = (id: number) => {
        router.push(`/dashboard/datasets/view?id=${id}`);
        setOpened(false);
    };

    const activeCount = downloads.filter(
        d => d.status === 'pending' || d.status === 'downloading'
    ).length;

    return (
        <Menu opened={opened} onChange={setOpened} position="bottom-end" width={350}>
            <Menu.Target>
                <Indicator disabled={activeCount === 0} processing={activeCount > 0} color="blue">
                    <ActionIcon
                        variant="subtle"
                        aria-label="Downloads"
                        onClick={() => setOpened(o => !o)}
                    >
                        <IconDownload style={{ width: 22, height: 22 }} />
                    </ActionIcon>
                </Indicator>
            </Menu.Target>

            <Menu.Dropdown>
                <Group justify="space-between" align="center" px="md" py={8}>
                    <Title order={6}>Dataset Downloads</Title>
                    {downloads.length > 0 && (
                        <ActionIcon
                            color="red"
                            variant="subtle"
                            onClick={clearCompleted}
                            disabled={!downloads.some(d => d.status === 'completed' || d.status === 'failed')}
                            title="Clear Completed"
                        >
                            <IconTrash size={16} />
                        </ActionIcon>
                    )}
                </Group>

                <Divider />

                {downloads.length === 0 ? (
                    <Text c="dimmed" size="sm" ta="center" py="md">No active downloads</Text>
                ) : (
                    <>
                        {downloads.map((download) => (
                            <Paper key={download.id} p="md" withBorder={false} mb={0}>
                                <Group position="apart" mb={0}>
                                    <Box style={{ flex: 1 }}>
                                        <Text
                                            size="sm"
                                            fw={500}
                                            style={{
                                                cursor: download.status === 'completed' ? 'pointer' : 'default',
                                                color: download.status === 'completed' ? '#228be6' : undefined,
                                                textDecoration: download.status === 'completed' ? 'underline' : 'none',
                                            }}
                                            onClick={() => {
                                                if (download.status === 'completed') {
                                                    handleViewDataset(download.id);
                                                }
                                            }}
                                        >
                                            {download.name}
                                        </Text>

                                        {(download.status === 'pending' || download.status === 'downloading') && (
                                            <>
                                                <Text size="xs" c="dimmed">Downloading...</Text>
                                                <Progress
                                                    value={download.progress}
                                                    size="sm"
                                                    mt={5}
                                                    color="blue"
                                                    striped
                                                    animated
                                                />
                                            </>
                                        )}

                                        {download.status === 'completed' && (
                                            <Text size="xs" c="dimmed">Download completed</Text>
                                        )}

                                        {download.status === 'failed' && (
                                            <Text size="xs" c="red">Download failed</Text>
                                        )}
                                    </Box>

                                    {download.status === 'completed' && (
                                        <ActionIcon color="green" variant="filled" radius="xl" size="md">
                                            <IconCheck size={16} />
                                        </ActionIcon>
                                    )}

                                    {download.status === 'failed' && (
                                        <ActionIcon color="red" variant="filled" radius="xl" size="md">
                                            <IconX size={16} />
                                        </ActionIcon>
                                    )}

                                    {(download.status === 'pending' || download.status === 'downloading') && (
                                        <Badge color="blue">Downloading</Badge>
                                    )}
                                </Group>
                                <Divider my={10} />
                            </Paper>
                        ))}
                    </>
                )}
            </Menu.Dropdown>
        </Menu>
    );
} 