'use client';

import React, { useState, useEffect } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import {
  Container,
  Grid,
  Card,
  Text,
  rem,
  Divider,
  Button,
  Group,
  Switch,
  useMantineTheme,
  Badge,
  ActionIcon,
  Highlight,
  CopyButton,
  Tooltip,
  Modal,
  Title,
  LoadingOverlay,
  Stack,
  Textarea,
  Skeleton,
} from '@mantine/core';
import {
  IconCopy,
  IconCheck,
  IconX,
  IconChevronLeft,
  IconChevronRight,
  IconMaximize,
  IconEdit,
  IconDeviceFloppy,
} from '@tabler/icons-react';
import ReactMarkdown from 'react-markdown';
import { notifications } from '@mantine/notifications';
import ReactDiffViewer from 'react-diff-viewer-continued';
import * as Diff from 'diff';

import InteractiveRecordModal from './expand';
import { AccordionStats } from './(components)/accordion';
import { SortButton } from './sort';
import SearchComponent from './search';
import CustomPagination from './pagination';

interface Record {
  [key: string]: string;
}

interface SplitInfo {
  name: string;
  count: number;
}

interface ApiResponse {
  metadata: {
    id: number;
    name: string;
    status: string;
  };
  data: Record[];
  pagination: {
    total: number;
    page: number;
    page_size: number;
    total_pages: number;
  };
  columns: string[];
  format: string;
  splits_info?: SplitInfo[];
  error?: string;
}

interface DatasetInfo {
  id: number;
  name: string;
  status: string;
  clustering_status: string;
  is_clustered: boolean;
  created_at: string;
  source: string;
  identifier: string;
}

const BACKEND_PAGE_SIZE = 10; // Fetch 100 records at a time from backend

const DatasetView: React.FC = () => {
  const theme = useMantineTheme();
  const searchParams = useSearchParams();
  const router = useRouter();
  const datasetId = searchParams.get('id');
  const subclusterId = searchParams.get('subcluster');

  const [renderMarkdown, setRenderMarkdown] = useState<boolean>(false);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [totalRecords, setTotalRecords] = useState<number>(0);
  const [allRecords, setAllRecords] = useState<Record[]>([]);
  const [filteredRecords, setFilteredRecords] = useState<Record[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerms, setSearchTerms] = useState<string[]>([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<{ key: string; value: string } | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [records, setRecords] = useState<Record[]>([]);
  const [isInitialLoading, setIsInitialLoading] = useState<boolean>(true);
  const [isPageLoading, setIsPageLoading] = useState<boolean>(false);
  const [currentBackendPage, setCurrentBackendPage] = useState<number>(1);
  const [cachedRecords, setCachedRecords] = useState<Record[]>([]);
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [editingFieldKey, setEditingFieldKey] = useState<string | null>(null);
  const [currentEditValue, setCurrentEditValue] = useState<string>('');
  const [originalEditValue, setOriginalEditValue] = useState<string>('');
  const [splitsInfo, setSplitsInfo] = useState<SplitInfo[] | null>(null);
  const [addedLinesCount, setAddedLinesCount] = useState<number>(0);
  const [deletedLinesCount, setDeletedLinesCount] = useState<number>(0);

  useEffect(() => {
    if (datasetId) {
      fetchDatasetInfo();
      if (subclusterId) {
        fetchSubclusterTexts();
      } else {
        fetchInitialRecords();
      }
    }
  }, [datasetId, subclusterId]);

  // Calculate which backend page we need based on the current frontend page
  useEffect(() => {
    const requiredBackendPage = Math.ceil(currentPage / BACKEND_PAGE_SIZE);
    if (requiredBackendPage !== currentBackendPage) {
      setCurrentBackendPage(requiredBackendPage);
    }
  }, [currentPage]);

  // Fetch backend page when it changes
  useEffect(() => {
    if (datasetId && !subclusterId && !isInitialLoading) {
      if (currentBackendPage > 1) {
        console.log(`Backend page changed to ${currentBackendPage}, fetching...`);
        fetchBackendPage(currentBackendPage);
      } else {
        console.log(`Backend page is ${currentBackendPage}, relying on initial load cache.`);
      }
    }
  }, [currentBackendPage, datasetId, subclusterId, isInitialLoading]);

  // Update displayed records when page changes
  useEffect(() => {
    if (cachedRecords && cachedRecords.length > 0) {
      // Calculate the index within the current backend page/chunk
      const indexInBackendPage = (currentPage - 1) % BACKEND_PAGE_SIZE;

      // Check if the required record exists in the current cache
      if (indexInBackendPage >= 0 && indexInBackendPage < cachedRecords.length) {
        const currentRecord = cachedRecords[indexInBackendPage];
        // Set only the single record for display
        setFilteredRecords([currentRecord]);
        setRecords([currentRecord]);
      } else {
        // Record not in cache, likely waiting for fetchBackendPage to update cache
        setFilteredRecords([]);
        setRecords([]);
        // The other useEffect hooks handle fetching the correct backend page
      }
    } else {
      // No cached records yet
      setFilteredRecords([]);
      setRecords([]);
    }
  }, [currentPage, cachedRecords]); // Rerun when page changes or cache updates

  // Calculate Diff Counts Effect
  useEffect(() => {
    if (isEditing && editingFieldKey) {
      try {
        // Check if Diff.diffLines exists and use it, otherwise use a simple fallback
        let added = 0;
        let removed = 0;

        if (typeof Diff.diffLines === 'function') {
          // Use the built-in diffLines if available
          const diffResult = Diff.diffLines(originalEditValue, currentEditValue);

          interface DiffPart {
            added?: boolean;
            removed?: boolean;
            count?: number;
            value: string;
          }

          diffResult.forEach((part: DiffPart) => {
            if (part.added) {
              added += part.count || 0;
            } else if (part.removed) {
              removed += part.count || 0;
            }
          });
        } else {
          // Simple fallback: split by lines and compare
          const originalLines = originalEditValue.split('\n');
          const currentLines = currentEditValue.split('\n');

          // Very simplified diff count - not accurate but prevents errors
          const originalLineCount = originalLines.length;
          const currentLineCount = currentLines.length;

          if (currentLineCount > originalLineCount) {
            added = currentLineCount - originalLineCount;
          } else if (originalLineCount > currentLineCount) {
            removed = originalLineCount - currentLineCount;
          }
        }

        setAddedLinesCount(added);
        setDeletedLinesCount(removed);
      } catch (error) {
        console.error('Error calculating diff:', error);
        setAddedLinesCount(0);
        setDeletedLinesCount(0);
      }
    } else {
      // Reset counts when not editing
      setAddedLinesCount(0);
      setDeletedLinesCount(0);
    }
  }, [isEditing, editingFieldKey, originalEditValue, currentEditValue]);

  const fetchDatasetInfo = async () => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}?detail_level=full`
      );
      if (!response.ok) {
        throw new Error('Failed to fetch dataset info');
      }
      const data = await response.json();
      setDatasetInfo(data);
    } catch (error) {
      console.error('Error fetching dataset info:', error);
      notifications.show({
        title: 'Error',
        message: 'Failed to fetch dataset information',
        color: 'red',
      });
      setLoading(false);
    }
  };

  const fetchInitialRecords = async () => {
    try {
      setIsInitialLoading(true);
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}?detail_level=data&page=1&page_size=${BACKEND_PAGE_SIZE}`
      );

      if (!response.ok) {
        const errorData = await response.json();
        let errorMessage = 'Failed to fetch records';

        switch (response.status) {
          case 404:
            errorMessage = `Dataset not found: ${errorData.detail}`;
            break;
          case 400:
            errorMessage = `Invalid request: ${errorData.detail}`;
            break;
          case 500:
            errorMessage = `Server error: ${errorData.detail}`;
            break;
          default:
            errorMessage = errorData.detail || errorMessage;
        }

        throw new Error(errorMessage);
      }

      const data: ApiResponse = await response.json();

      // Check if data and data.data exist
      if (!data || !data.data) {
        throw new Error('Invalid response format: records data is missing');
      }

      setCachedRecords(data.data);
      setTotalRecords(data.pagination.total);
      setSplitsInfo(data.splits_info || null);

      // Show only the first record initially (currentPage is 1)
      if (data.data && data.data.length > 0) {
        setFilteredRecords([data.data[0]]);
        setRecords([data.data[0]]);
      } else {
        setFilteredRecords([]);
        setRecords([]);
      }
    } catch (error: any) {
      setError(error.message);
      notifications.show({
        title: 'Error',
        message: error.message,
        color: 'red',
      });
      // Set empty arrays to prevent undefined errors
      setCachedRecords([]);
      setFilteredRecords([]);
      setRecords([]);
      setSplitsInfo(null);
    } finally {
      setIsInitialLoading(false);
    }
  };

  const fetchBackendPage = async (backendPage: number) => {
    if (backendPage < 1) {
      console.error('Attempted to fetch backend page < 1 (', backendPage, '). Aborting fetch.');
      setError('Invalid page number requested.');
      setIsPageLoading(false);
      return;
    }

    console.log('Fetching backend page:', backendPage);
    try {
      setIsPageLoading(true);
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}?detail_level=data&page=${backendPage}&page_size=${BACKEND_PAGE_SIZE}`
      );

      if (!response.ok) {
        const errorData = await response.json();
        let errorMessage = 'Failed to fetch records';

        switch (response.status) {
          case 404:
            errorMessage = `Dataset not found: ${errorData.detail}`;
            break;
          case 400:
            errorMessage = `Invalid request: ${errorData.detail}`;
            break;
          case 500:
            errorMessage = `Server error: ${errorData.detail}`;
            break;
          default:
            errorMessage = errorData.detail || errorMessage;
        }

        throw new Error(errorMessage);
      }

      const data: ApiResponse = await response.json();

      // Check if data and data.data exist
      if (!data || !data.data) {
        throw new Error('Invalid response format: records data is missing');
      }

      // Update the cache with the new chunk of records
      setCachedRecords(data.data);
      setTotalRecords(data.pagination.total);
      setSplitsInfo(data.splits_info || null);

      // Trigger record update based on currentPage and new cache
      // The useEffect watching [currentPage, cachedRecords] will handle this.
    } catch (error: any) {
      setError(error.message);
      notifications.show({
        title: 'Error',
        message: error.message,
        color: 'red',
      });
      // Set empty arrays to prevent undefined errors
      setCachedRecords([]);
      setFilteredRecords([]);
      setRecords([]);
      setSplitsInfo(null);
    } finally {
      setIsPageLoading(false);
    }
  };

  const fetchSubclusterTexts = async () => {
    try {
      setIsInitialLoading(true);
      setLoading(true);
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/level1_clusters/${subclusterId}/texts`
      );

      if (!response.ok) {
        const errorData = await response.json();
        let errorMessage = 'Failed to fetch subcluster texts';

        switch (response.status) {
          case 404:
            errorMessage = `Subcluster not found: ${errorData.detail}`;
            break;
          case 500:
            errorMessage = `Server error: ${errorData.detail}`;
            break;
          default:
            errorMessage = errorData.detail || errorMessage;
        }

        throw new Error(errorMessage);
      }

      const data = await response.json();
      const formattedRecords = data.texts.map((text: any) => ({
        text: text.text,
      }));
      setRecords(formattedRecords);
      setFilteredRecords(formattedRecords);
      setTotalRecords(formattedRecords.length);
      setCachedRecords(formattedRecords);
    } catch (error: any) {
      setError(error.message);
      notifications.show({
        title: 'Error',
        message: error.message,
        color: 'red',
      });
    } finally {
      setIsInitialLoading(false);
      setLoading(false);
    }
  };

  const handleSearch = async (term: string) => {
    const terms = term.split(' ').filter((t) => t.trim() !== '');
    setSearchTerms(terms);

    if (terms.length === 0) {
      fetchInitialRecords();
      return;
    }

    try {
      setIsInitialLoading(true);
      const searchParams = new URLSearchParams({
        page: '1',
        page_size: BACKEND_PAGE_SIZE.toString(),
        search: terms.join(' '),
      });

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/datasets/${datasetId}/records?${searchParams}`
      );
      if (!response.ok) {
        throw new Error('Failed to search records');
      }
      const data: ApiResponse = await response.json();
      setCachedRecords(data.data);
      setTotalRecords(data.pagination.total);
      setCurrentPage(1);

      // Show the first record of the search results
      if (data.data && data.data.length > 0) {
        setFilteredRecords([data.data[0]]);
        setRecords([data.data[0]]);
      } else {
        setFilteredRecords([]);
        setRecords([]);
      }
    } catch (error: any) {
      notifications.show({
        title: 'Error',
        message: error.message || 'Failed to search records',
        color: 'red',
      });
    } finally {
      setIsInitialLoading(false);
    }
  };

  const preStyles: React.CSSProperties = {
    fontFamily: 'inherit',
    fontWeight: 300,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    width: '100%',
    margin: 0,
    padding: theme.spacing.sm,
    backgroundColor: 'lavenderblush',
    borderRadius: theme.radius.md,
    border: '1px solid black',
  };

  const customRenderers = {
    p: ({ node, children }: any) => {
      const content = Array.isArray(children) ? children.join('') : children;
      return <Highlight highlight={searchTerms}>{content}</Highlight>;
    },
  };

  // Helper function to safely render any value as a string
  const safeString = (value: any): string => {
    if (value === null || value === undefined) {
      return '';
    }
    if (typeof value === 'string') {
      return value;
    }
    if (typeof value === 'object') {
      try {
        return JSON.stringify(value);
      } catch (e) {
        console.error('Error stringifying object value:', e);
        return '[Object]';
      }
    }
    return String(value);
  };

  const getStructuredFields = (record: Record) => {
    const fields: { key: string; value: string }[] = [];
    let index = 1;

    while (record[`structured_key_${index}`]) {
      fields.push({
        key: safeString(record[`structured_key_${index}`]),
        value: safeString(record[`structured_value_${index}`]),
      });
      index++;
    }

    return fields;
  };

  const getOriginalFields = (record: Record) =>
    Object.entries(record)
      .filter(([key]) => !key.startsWith('structured_key_') && !key.startsWith('structured_value_'))
      .map(([key, value]) => [key, safeString(value)]);

  // Add save function
  const handleSaveEdit = async () => {
    // In a real implementation, you would save to your backend
    // For now, just update the UI and show notification
    if (editingFieldKey && filteredRecords.length > 0) {
      const newRecords = [...filteredRecords];
      const record = { ...newRecords[0] };

      // Update the record with edited content
      record[editingFieldKey] = currentEditValue;
      newRecords[0] = record;

      setFilteredRecords(newRecords);

      // Also update in cached records
      const indexInBackendPage = (currentPage - 1) % BACKEND_PAGE_SIZE;
      if (cachedRecords.length > indexInBackendPage) {
        const newCachedRecords = [...cachedRecords];
        newCachedRecords[indexInBackendPage] = record;
        setCachedRecords(newCachedRecords);
      }

      notifications.show({
        title: 'Success',
        message: `Updated ${editingFieldKey} field`,
        color: 'green',
      });

      // Exit edit mode
      setIsEditing(false);
      setEditingFieldKey(null);
    }
  };

  // Function to start editing a field
  const startEditing = (key: string, value: string) => {
    setEditingFieldKey(key);
    setOriginalEditValue(value);
    setCurrentEditValue(value);
    setIsEditing(true);
    // Reset counts immediately on starting edit (will be recalculated by effect)
    setAddedLinesCount(0);
    setDeletedLinesCount(0);
  };

  // Function to cancel editing
  const cancelEditing = () => {
    setIsEditing(false);
    setEditingFieldKey(null);
    setOriginalEditValue('');
    setCurrentEditValue('');
    // Reset counts on cancel
    setAddedLinesCount(0);
    setDeletedLinesCount(0);
  };

  // Placeholder for future split filtering
  const handleSplitClick = (splitName: string) => {
    console.log(`Filtering by split: ${splitName}`);
    notifications.show({
      title: 'Filter Action',
      message: `Clicked on split: ${splitName}. Filtering not yet implemented.`,
      color: 'blue',
    });
    // TODO: Implement actual filtering logic here
    // This might involve:
    // - Setting a filter state variable
    // - Re-fetching data from the backend with a split filter parameter
    // - Or filtering the `cachedRecords` directly on the frontend (if feasible)
  };

  return (
    <Container my="md" fluid>
      <LoadingOverlay visible={isInitialLoading} overlayProps={{ blur: 2 }} />
      <Card shadow="sm" radius="md" withBorder mb="md">
        <Group justify="space-between" mb="md">
          <div>
            <Title order={2}>{datasetInfo?.name || 'Loading...'}</Title>
            <Text c="dimmed" size="sm">
              {datasetInfo?.identifier ? `Identifier: ${datasetInfo.identifier}` : 'No identifier'}{' '}
              • {datasetInfo?.status ? `Status: ${datasetInfo.status}` : 'No status'}
              {subclusterId && ' • Viewing Subcluster Texts'}
            </Text>
          </div>
          <Group>
            {!subclusterId && (
              <Button
                variant="filled"
                color="black"
                radius="md"
                onClick={() => router.push(`/dashboard/datasets/cluster?id=${datasetId}`)}
                styles={{
                  root: {
                    transition: 'background-color 0.2s ease',
                    '&:hover': {
                      backgroundColor: '#333',
                    },
                  },
                }}
              >
                View Clusters
              </Button>
            )}
            <Button
              variant="light"
              color="black"
              radius="md"
              onClick={() => router.push('/dashboard/datasets')}
              styles={{
                root: {
                  transition: 'background-color 0.2s ease',
                  '&:hover': {
                    backgroundColor: '#f0f0f0',
                  },
                },
              }}
            >
              Back to Datasets
            </Button>
          </Group>
        </Group>
      </Card>

      <Grid gutter="md">
        <Grid.Col span={{ base: 12, md: 8 }}>
          <Card shadow="sm" radius="md" withBorder pb={50}>
            <Card.Section
              withBorder
              inheritPadding
              py="xs"
              style={{
                backgroundColor: 'lavenderblush',
                opacity: isPageLoading ? 0.7 : 1,
                transition: 'opacity 0.2s ease',
              }}
            >
              <Group justify="space-between" align="center">
                <Text fw={300} size="sm">
                  Dataset Record {isPageLoading && '(Loading...)'}
                </Text>
                <Group gap="xs">
                  <CustomPagination
                    currentPage={currentPage}
                    totalPages={totalRecords} // Display total records as total pages
                    onPageChange={setCurrentPage}
                    disabled={isPageLoading || isEditing}
                  />

                  {isEditing && (
                    <Group gap="xs">
                      <ActionIcon color="red" variant="light" onClick={cancelEditing}>
                        <IconX size="1rem" />
                      </ActionIcon>
                      <ActionIcon color="green" variant="light" onClick={handleSaveEdit}>
                        <IconDeviceFloppy size="1rem" />
                      </ActionIcon>
                    </Group>
                  )}

                  <Switch
                    checked={renderMarkdown}
                    onChange={(event) => setRenderMarkdown(event.currentTarget.checked)}
                    color="teal"
                    size="sm"
                    label="Markdown"
                    disabled={isEditing}
                    thumbIcon={
                      renderMarkdown ? (
                        <IconCheck
                          style={{ width: rem(12), height: rem(12) }}
                          color={theme.colors.teal[6]}
                          stroke={3}
                        />
                      ) : (
                        <IconX
                          style={{ width: rem(12), height: rem(12) }}
                          color={theme.colors.red[6]}
                          stroke={3}
                        />
                      )
                    }
                  />
                </Group>
              </Group>
            </Card.Section>

            <Card.Section inheritPadding mt="md">
              {error ? (
                <Text c="red">{error}</Text>
              ) : isEditing && editingFieldKey ? (
                <div>
                  <Group justify="space-between" mb="xs">
                    <Badge color="blue">Editing: {editingFieldKey}</Badge>
                  </Group>

                  <Textarea
                    placeholder={`Edit ${editingFieldKey}...`}
                    value={currentEditValue}
                    onChange={(event) => setCurrentEditValue(event.currentTarget.value)}
                    minRows={10}
                    maxRows={20}
                    autosize
                    mb="md"
                    styles={{ input: { fontFamily: 'monospace' } }}
                  />

                  {/* Diff View Title with Counts */}
                  <Group mb="xs" gap="xs" align="center">
                    <Text size="sm" fw={500}>
                      Diff View:
                    </Text>
                    {addedLinesCount > 0 && (
                      <Text size="sm" c="green" fw={500}>
                        +{addedLinesCount}
                      </Text>
                    )}
                    {deletedLinesCount > 0 && (
                      <Text size="sm" c="red" fw={500}>
                        -{deletedLinesCount}
                      </Text>
                    )}
                  </Group>
                  <ReactDiffViewer
                    oldValue={originalEditValue}
                    newValue={currentEditValue}
                    splitView
                    showDiffOnly={false}
                    hideLineNumbers={false}
                    useDarkTheme={false}
                  />
                </div>
              ) : filteredRecords.length > 0 ? (
                <>
                  {(() => {
                    const record = filteredRecords[0]; // Get the single record for the current page
                    return (
                      <>
                        {getOriginalFields(record).map(([fieldKey, fieldValue], fieldIndex) => (
                          <React.Fragment key={fieldKey}>
                            <Group justify="space-between" mb="xs">
                              <Badge color={fieldKey === 'chat' ? 'blue' : 'gray'}>
                                {fieldKey}
                              </Badge>
                              <Group gap="xs">
                                <CopyButton value={fieldValue} timeout={2000}>
                                  {({ copied, copy }) => (
                                    <Tooltip
                                      label={copied ? 'Copied' : 'Copy'}
                                      withArrow
                                      position="right"
                                    >
                                      <ActionIcon
                                        color={copied ? 'teal' : 'gray'}
                                        variant="subtle"
                                        onClick={copy}
                                      >
                                        {copied ? (
                                          <IconCheck style={{ width: rem(16) }} />
                                        ) : (
                                          <IconCopy style={{ width: rem(16) }} />
                                        )}
                                      </ActionIcon>
                                    </Tooltip>
                                  )}
                                </CopyButton>
                                <Tooltip label="Edit" withArrow position="right">
                                  <ActionIcon
                                    color="orange"
                                    variant="subtle"
                                    onClick={() => startEditing(fieldKey, fieldValue)}
                                    disabled={isEditing}
                                  >
                                    <IconEdit style={{ width: rem(16) }} />
                                  </ActionIcon>
                                </Tooltip>
                                <Tooltip label="Expand" withArrow position="right">
                                  <ActionIcon
                                    color="blue"
                                    variant="subtle"
                                    onClick={() => {
                                      setSelectedRecord({ key: fieldKey, value: fieldValue });
                                      setModalOpen(true);
                                    }}
                                  >
                                    <IconMaximize style={{ width: rem(16) }} />
                                  </ActionIcon>
                                </Tooltip>
                              </Group>
                            </Group>
                            <pre style={preStyles}>
                              <Highlight highlight={searchTerms}>{fieldValue}</Highlight>
                            </pre>
                            {fieldIndex < getOriginalFields(record).length - 1 && (
                              <Divider my="sm" />
                            )}
                          </React.Fragment>
                        ))}

                        {getStructuredFields(record).length > 0 && (
                          <>
                            <Divider my="sm" label="Structured Content" labelPosition="center" />
                            <div style={{ paddingLeft: rem(20) }}>
                              {getStructuredFields(record).map((field, structuredIndex) => (
                                <React.Fragment key={structuredIndex}>
                                  <Group gap="xs" mb="xs">
                                    <Group gap={4}>
                                      <Text color="dimmed" size="sm">
                                        └─
                                      </Text>
                                      <Badge
                                        color="pink"
                                        variant="light"
                                        style={{ cursor: 'pointer' }}
                                        onClick={() => {
                                          setSelectedRecord({ key: field.key, value: field.value });
                                          setModalOpen(true);
                                        }}
                                      >
                                        {field.key}
                                      </Badge>
                                    </Group>
                                    <Group gap="xs">
                                      <CopyButton value={field.value} timeout={2000}>
                                        {({ copied, copy }) => (
                                          <Tooltip
                                            label={copied ? 'Copied' : 'Copy'}
                                            withArrow
                                            position="right"
                                          >
                                            <ActionIcon
                                              color={copied ? 'teal' : 'gray'}
                                              variant="subtle"
                                              onClick={copy}
                                            >
                                              {copied ? (
                                                <IconCheck style={{ width: rem(16) }} />
                                              ) : (
                                                <IconCopy style={{ width: rem(16) }} />
                                              )}
                                            </ActionIcon>
                                          </Tooltip>
                                        )}
                                      </CopyButton>
                                      <Tooltip label="Edit" withArrow position="right">
                                        <ActionIcon
                                          color="orange"
                                          variant="subtle"
                                          onClick={() => startEditing(field.key, field.value)}
                                          disabled={isEditing}
                                        >
                                          <IconEdit style={{ width: rem(16) }} />
                                        </ActionIcon>
                                      </Tooltip>
                                      <Tooltip label="Expand" withArrow position="right">
                                        <ActionIcon
                                          color="blue"
                                          variant="subtle"
                                          onClick={() => {
                                            setSelectedRecord({
                                              key: field.key,
                                              value: field.value,
                                            });
                                            setModalOpen(true);
                                          }}
                                        >
                                          <IconMaximize style={{ width: rem(16) }} />
                                        </ActionIcon>
                                      </Tooltip>
                                    </Group>
                                  </Group>
                                  <div style={{ paddingLeft: rem(35) }}>
                                    {renderMarkdown ? (
                                      <ReactMarkdown components={customRenderers}>
                                        {field.value}
                                      </ReactMarkdown>
                                    ) : (
                                      <pre style={preStyles}>
                                        <Highlight highlight={searchTerms}>{field.value}</Highlight>
                                      </pre>
                                    )}
                                  </div>
                                  {structuredIndex < getStructuredFields(record).length - 1 && (
                                    <Divider my="sm" variant="dashed" />
                                  )}
                                </React.Fragment>
                              ))}
                            </div>
                          </>
                        )}
                      </>
                    );
                  })()}
                </>
              ) : (
                <Text>No records found</Text>
              )}
            </Card.Section>
          </Card>
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 4 }}>
          <Card shadow="sm" radius="md" withBorder>
            <Card.Section withBorder inheritPadding py="xs">
              <Text fw={500}>Dataset Details</Text>
            </Card.Section>
            <Card.Section inheritPadding mt="md" pb="md">
              {datasetInfo ? (
                <Stack gap="xs">
                  <Group justify="space-between">
                    <Text size="sm" fw={500}>
                      Identifier:
                    </Text>
                    <Text size="sm">{datasetInfo.identifier}</Text>
                  </Group>
                  <Group justify="space-between">
                    <Text size="sm" fw={500}>
                      Source:
                    </Text>
                    <Text size="sm">{datasetInfo.source}</Text>
                  </Group>
                  <Group justify="space-between">
                    <Text size="sm" fw={500}>
                      Status:
                    </Text>
                    <Badge
                      color={datasetInfo.status === 'completed' ? 'green' : 'yellow'}
                      size="sm"
                    >
                      {datasetInfo.status}
                    </Badge>
                  </Group>
                  <Group justify="space-between">
                    <Text size="sm" fw={500}>
                      Clustering:
                    </Text>
                    <Badge
                      color={
                        datasetInfo.clustering_status === 'completed'
                          ? 'blue'
                          : datasetInfo.is_clustered
                            ? 'cyan'
                            : 'gray'
                      }
                      size="sm"
                    >
                      {datasetInfo.clustering_status} {datasetInfo.is_clustered ? '(Active)' : ''}
                    </Badge>
                  </Group>
                  <Group justify="space-between">
                    <Text size="sm" fw={500}>
                      Created:
                    </Text>
                    <Text size="sm">{new Date(datasetInfo.created_at).toLocaleString()}</Text>
                  </Group>
                  {splitsInfo && splitsInfo.length > 0 && (
                    <Group justify="space-between" align="flex-start">
                      <Text size="sm" fw={500}>
                        Splits:
                      </Text>
                      <Group gap={4} wrap="wrap" justify="flex-end" style={{ maxWidth: '60%' }}>
                        {splitsInfo.map((split) => (
                          <Badge
                            key={split.name}
                            variant="outline"
                            size="sm"
                            onClick={() => handleSplitClick(split.name)}
                            style={{ cursor: 'pointer' }}
                            title={`Click to filter by ${split.name}`}
                          >
                            {split.name} ({split.count.toLocaleString()})
                          </Badge>
                        ))}
                      </Group>
                    </Group>
                  )}
                  <Group justify="space-between">
                    <Text size="sm" fw={500}>
                      ID:
                    </Text>
                    <Text size="sm">{datasetInfo.id}</Text>
                  </Group>
                </Stack>
              ) : (
                <Text size="sm" c="dimmed">
                  Loading details...
                </Text>
              )}
            </Card.Section>
          </Card>
        </Grid.Col>
      </Grid>

      {selectedRecord && (
        <Modal
          opened={modalOpen}
          onClose={() => setModalOpen(false)}
          size="xl"
          title={selectedRecord.key}
        >
          <InteractiveRecordModal
            content={selectedRecord.value}
            onClose={() => setModalOpen(false)}
          />
        </Modal>
      )}
    </Container>
  );
};

export default DatasetView;
