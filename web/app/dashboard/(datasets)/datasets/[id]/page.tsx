'use client';

import { Button, Group, Text } from "@mantine/core";
import { IconRefresh } from "@tabler/icons-react";
import { useRouter, useParams } from "next/navigation";
import { useState, useEffect } from "react";

const DatasetView = () => {
    const router = useRouter();
    const params = useParams();
    const id = params.id;
    const [loading, setLoading] = useState(false);

    const fetchData = () => {
        setLoading(true);
        // Fetch data logic would go here
        setTimeout(() => {
            setLoading(false);
        }, 500);
    };

    return (
        <div className="p-6">
            <Group justify="apart" mb="md">
                <Text size="xl" fw={700}>
                    Dataset Details
                </Text>
                <Group>
                    <Button
                        variant="light"
                        color="blue"
                        onClick={() => router.push(`/dashboard/datasets/cluster?id=${id}`)}
                    >
                        View Clusters
                    </Button>
                    <Button
                        leftSection={<IconRefresh size={16} />}
                        onClick={fetchData}
                        loading={loading}
                    >
                        Refresh
                    </Button>
                </Group>
            </Group>

      // ... rest of the existing code ...
        </div>
    );
};

export default DatasetView; 