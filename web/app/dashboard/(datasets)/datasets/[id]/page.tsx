'use client';

import { Button, Group, Text, Box, Skeleton, Textarea, Paper } from '@mantine/core';
import { IconRefresh } from '@tabler/icons-react';
import { useRouter, useParams } from 'next/navigation';
import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import ReactDiffViewer from 'react-diff-viewer-continued';

const DatasetView = () => {
  const router = useRouter();
  const params = useParams();
  const { id } = params;
  const [loading, setLoading] = useState(false);
  const [markdownContent, setMarkdownContent] = useState('');
  const [originalMarkdownContent, setOriginalMarkdownContent] = useState('');
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    fetchData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  const fetchData = () => {
    setLoading(true);
    setTimeout(() => {
      const fetchedMarkdown = `# Dataset: ${id}\n\nThis is some initial **markdown** content for the dataset.\n\n- Detail 1\n- Detail 2\n\nMore descriptive text goes here.`;
      setMarkdownContent(fetchedMarkdown);
      setOriginalMarkdownContent(fetchedMarkdown);
      setLoading(false);
    }, 500);
  };

  const handleEdit = () => {
    setIsEditing(true);
  };

  const handleCancel = () => {
    setMarkdownContent(originalMarkdownContent);
    setIsEditing(false);
  };

  const handleSave = () => {
    console.log('Saving dataset:', id, 'Content:', markdownContent);
    setOriginalMarkdownContent(markdownContent);
    setIsEditing(false);
  };

  return (
    <div className="p-6">
      <Group justify="space-between" mb="md">
        <Text size="xl" fw={700}>
          Dataset Details {id}
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
            loading={loading && !isEditing}
          >
            Refresh
          </Button>
        </Group>
      </Group>

      <Box mt="lg">
        <Group justify="space-between" mb="sm">
          <Text fw={500}>Description</Text>
          {!isEditing ? (
            <Button variant="outline" size="xs" onClick={handleEdit} disabled={loading}>
              Edit
            </Button>
          ) : (
            <Group>
              <Button variant="outline" size="xs" color="gray" onClick={handleCancel}>
                Cancel
              </Button>
              <Button variant="filled" size="xs" onClick={handleSave}>
                Save
              </Button>
            </Group>
          )}
        </Group>

        <Paper shadow="xs" p="md" withBorder>
          {loading ? (
            <Skeleton height={150} radius="sm" />
          ) : isEditing ? (
            <div>
              <Textarea
                placeholder="Enter markdown description..."
                value={markdownContent}
                onChange={(event) => setMarkdownContent(event.currentTarget.value)}
                minRows={10}
                autosize
                styles={{ input: { fontFamily: 'monospace' } }}
                mb="md"
              />
              <ReactDiffViewer
                oldValue={originalMarkdownContent}
                newValue={markdownContent}
                splitView
                showDiffOnly={false}
                hideLineNumbers={false}
              />
            </div>
          ) : (
            <div>
              {markdownContent ? (
                <ReactMarkdown className="prose prose-sm max-w-none">
                  {markdownContent}
                </ReactMarkdown>
              ) : (
                <Text c="dimmed">No description available.</Text>
              )}
            </div>
          )}
        </Paper>
      </Box>
    </div>
  );
};

export default DatasetView;
