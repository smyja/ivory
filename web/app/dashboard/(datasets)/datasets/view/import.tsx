'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { z } from 'zod';
import { useForm, zodResolver } from '@mantine/form';
import classes from '@styles/global.module.css';
import { SelectOptionComponent } from '@/components/SelectOption/SelectOptionComponent';
import {
  TextInput,
  Paper,
  Title,
  Text,
  Container,
  Group,
  Button,
} from '@mantine/core';

interface FormValues {
  dataSource: string;
  datasetName: string;
  token: string;
}

const schema = z.object({
  dataSource: z.string().min(1, { message: 'Data source is required' }),
  datasetName: z.string().min(1, { message: 'Dataset name is required' }),
  token: z.string().optional(),
});

export default function AuthenticationTitle({ onClose }: { onClose: () => void }) {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const form = useForm<FormValues>({
    validate: zodResolver(schema),
    initialValues: {
      dataSource: 'HuggingFace',
      datasetName: '',
      token: '',
    },
  });

  const handleSubmit = async (values: FormValues) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/datasets', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(values),
      });

      if (!response.ok) {
        throw new Error('Failed to create dataset');
      }

      const result = await response.json();
      console.log('Dataset created:', result);
      router.push('/dashboard');
    } catch (error) {
      console.error('Failed to create dataset:', error);
      setError('Failed to create dataset. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={form.onSubmit(handleSubmit)}>
        <Container size={520} my={40}>
          <Title ta="center" className={classes.h1}>
            Create a Dataset
          </Title>

          <Text size="sm" ta="center" mt={5}>
            Import Data from numerous Data sources and start exploring
          </Text>

          <Paper withBorder shadow="md" p={30} mt={30} radius="md">
            <SelectOptionComponent
              value={form.values.dataSource}
              onChange={(value) => form.setFieldValue('dataSource', value)}
              disabledOptions={['CSV', 'Parquet']}
            />
            <TextInput
              {...form.getInputProps('datasetName')}
              label="Dataset Name"
              placeholder="Enter your Dataset name (username/dataset)"
              pt={10}
            />

            <TextInput
              {...form.getInputProps('token')}
              pt={10}
              label="Token (optional)"
              placeholder="Enter your token"
            />

            <Group justify="space-between" mt="md">
              {/* Add any additional options here if needed */}
            </Group>
            <Button fullWidth mt="xl" type="submit" loading={isLoading}>
              Load dataset
            </Button>

            {error && <Text c="red" mt="sm">{error}</Text>}
          </Paper>
        </Container>
      </form>
    </div>
  );
}