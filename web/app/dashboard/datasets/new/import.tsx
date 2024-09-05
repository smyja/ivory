'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { z } from 'zod';
import { useForm, zodResolver } from '@mantine/form';
import classes from '@styles/global.module.css';
import { SelectOptionComponent } from '@/components/SelectOption/SelectOptionComponent';
import {
  TextInput,
  PasswordInput,
  Checkbox,
  Anchor,
  Paper,
  Title,
  Text,
  Container,
  Group,
  Button,
  Divider,
} from '@mantine/core';

interface FormValues {
  email: string;
  password: string;
  datasetName: string;
  token: string;
}

export default function AuthenticationTitle({ onClose }: { onClose: () => void }) {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  const schema = z.object({
    email: z.string().email({ message: 'Invalid email' }),
    password: z.string().min(6, { message: 'Password should be at least 6 characters long' }),
    datasetName: z.string().min(1, { message: 'Dataset name is required' }),
    token: z.string().optional(),
  });

  const form = useForm<FormValues>({
    validate: zodResolver(schema),
    initialValues: {
      email: '',
      password: '',
      datasetName: '',
      token: '',
    },
  });

  const handleSubmit = async (values: FormValues) => {
    setIsLoading(true);
    setLoginError(null);

    try {
      // Here you would typically make an API call to create the dataset
      // For now, we'll just simulate a delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      console.log('Dataset created:', values);
      // After successful creation, you might want to redirect or show a success message
      router.push('/dashboard'); // Adjust this as needed
    } catch (error) {
      console.error('Failed to create dataset:', error);
      setLoginError('Failed to create dataset. Please try again.');
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
            <SelectOptionComponent />
            <TextInput
              {...form.getInputProps('datasetName')}
              label="Dataset Name"
              placeholder="Enter your Dataset name(username/dataset)"
              pt={10}
            />

            <TextInput
              {...form.getInputProps('token')}
              pt={10}
              label="Token (optional)"
              placeholder="Enter your token"
            />

      

            <Group justify="space-between" mt="md">
              <Checkbox label="Remember me" />
              <Anchor href="forgot-password" size="sm">
                Forgot password?
              </Anchor>
            </Group>
            <Button fullWidth mt="xl" type="submit" loading={isLoading}>
              Load dataset
            </Button>

            {loginError && <Text color="red" mt="sm">{loginError}</Text>}
          </Paper>
        </Container>
      </form>
    </div>
  );
}