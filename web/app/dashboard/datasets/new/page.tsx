'use client'
import React, { useState, useEffect } from 'react';
import { Container, Grid, Card, Text, rem, Divider, Button, Group, Switch, useMantineTheme, Badge, ActionIcon, TextInput } from '@mantine/core';
import ReactMarkdown from 'react-markdown';
import { IconCopy, IconCheck, IconX, IconChevronLeft, IconChevronRight, IconSearch } from '@tabler/icons-react';
import { AccordionStats } from '../accordion';
import { SortButton } from './sort';


interface Record {
    [key: string]: string;
  }
  
interface ApiResponse {
split: string;
total_records: number;
page: number;
page_size: number;
records: Record[];
}

interface CopyButtonProps {
  text: string;
}

const CopyButton: React.FC<CopyButtonProps> = ({ text }) => {
  const [copied, setCopied] = useState<boolean>(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <Button 
      onClick={handleCopy} 
      size="sm" 
      variant="subtle" 
      color={copied ? 'teal' : 'gray'}
      px={0}
    >
      {copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
    </Button>
  );
}

interface CustomPaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

const CustomPagination: React.FC<CustomPaginationProps> = ({ currentPage, totalPages, onPageChange }) => {
  return (
    <Group gap={5} align="center">
      <ActionIcon 
        color='gray'
        onClick={() => onPageChange(Math.max(1, currentPage - 1))}
        disabled={currentPage === 1}
      >
        <IconChevronLeft size="1rem" />
      </ActionIcon>
      <Text size="sm">
        {currentPage} of {totalPages.toLocaleString()}
      </Text>
      <ActionIcon 
        color='gray'
        onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
        disabled={currentPage === totalPages}
      >
        <IconChevronRight size="1rem" />
      </ActionIcon>
    </Group>
  );
}

const LeadGrid: React.FC = () => {
    const theme = useMantineTheme();
    const [renderMarkdown, setRenderMarkdown] = useState<boolean>(false);
    const [currentPage, setCurrentPage] = useState<number>(1);
    const [totalRecords, setTotalRecords] = useState<number>(0);
    const [allRecords, setAllRecords] = useState<Record[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
  
    useEffect(() => {
      fetchData();
    }, []);
  
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await fetch(`http://0.0.0.0:8000/datasets/7/records`);
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data: ApiResponse = await response.json();
        setAllRecords(data.records);
        setTotalRecords(data.total_records);
        setError(null);
      } catch (error) {
        setError('Failed to fetch data');
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };
  
    const currentRecord = allRecords[currentPage - 1] || null;
  
    const totalPages = totalRecords;
  
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
  
    return (
      <>   
        <Group mt={10} justify="space-between">  
          <TextInput
            placeholder="Search the dataset"
            style={{width: "600px"}}
            leftSection={<IconSearch style={{ width: rem(16), height: rem(16) }} stroke={1.5} />}
            value={""}
          />
          <Group justify="flex-end" gap="xs">
            <ActionIcon color='gray'>
              <IconChevronLeft size="1rem" />
            </ActionIcon>
            <Button variant="default">Filter</Button>
            <SortButton/>
          </Group>
        </Group>
        <Container my="md" fluid>
          <Grid gutter="md">
            <Grid.Col span={{ base: 12, md: 8 }}>
              <Card shadow="sm" radius="md" withBorder pb={50}>
                <Card.Section withBorder inheritPadding py="xs"  style={{backgroundColor:"lavenderblush"}}>
                  <Group justify="space-between" align="center">
                    <Text fw={300} size="sm">Dataset Record</Text>
                    <CustomPagination 
                      currentPage={currentPage}
                      totalPages={totalPages}
                      onPageChange={setCurrentPage}
                    />
                    <Switch
                      checked={renderMarkdown}
                      onChange={(event) => setRenderMarkdown(event.currentTarget.checked)}
                      color="teal"
                      size="sm"
                      label="Markdown"
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
                </Card.Section>
  
                <Card.Section inheritPadding mt="md">
                  {loading ? (
                    <Text>Loading...</Text>
                  ) : error ? (
                    <Text color="red">{error}</Text>
                  ) : currentRecord ? (
                    Object.entries(currentRecord).map(([key, value], index) => (
                      <React.Fragment key={index}>
                        <Group justify="space-between" mb="xs">
                          <Badge color="rgba(255, 110, 110, 1)">{key}</Badge>
                          <CopyButton text={value} />
                        </Group>
                        {renderMarkdown ? (
                          <ReactMarkdown>{value}</ReactMarkdown>
                        ) : (
                          <pre style={preStyles}>{value}</pre>
                        )}
                        {index < Object.entries(currentRecord).length - 1 && <Divider my="md" />}
                      </React.Fragment>
                    ))
                  ) : (
                    <Text>No record found</Text>
                  )}
                </Card.Section>
              </Card>
            </Grid.Col>
            <Grid.Col span={{ base: 12, md: 4 }}>
              <AccordionStats />
            </Grid.Col>
          </Grid>
        </Container>
      </>
    );
  }
  
  export default LeadGrid;