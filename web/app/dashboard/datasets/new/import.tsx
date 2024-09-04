'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { z } from 'zod';
import { useForm, zodResolver } from '@mantine/form';
import { useLogin } from '@/hooks';
import { useDispatch } from 'react-redux';
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
}

export default function AuthenticationTitle({ onClose }: { onClose: () => void }) {
  const schema = z.object({
    email: z.string().email({ message: 'Invalid email' }),
  });

  const form = useForm<FormValues>({
    validate: zodResolver(schema),
    initialValues: {
      email: '',
      password: '',
    },
  });

  const { setValue, isLoading, onChange, onSubmit, loginError } = useLogin({
    setValue: form.setValues,
    values: form.values,
  });
  return (
    <div>
      <form onSubmit={form.onSubmit(onSubmit)}>
        <Container size={520} my={40} >
          <Title ta="center" className={classes.h1}>
            Create a Dataset
          </Title>

          <Text size="sm" ta="center" mt={5}>
            Import Data from numerous Data sources and start exploring
           
          </Text>

          <Paper withBorder shadow="md" p={30} mt={30} radius="md">
            <SelectOptionComponent />
            <TextInput
              type="text"
              {...form.getInputProps('text')}
              label="Dataset Name"
              placeholder="Enter your Dataset name(username/dataset)"
              pt={10}
            />

            <TextInput
              type="text"
              pt={10}
              label="Token (optional)"
              placeholder="Enter your email"
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

            {loginError && <p>Failed to log in</p>}
          </Paper>
        </Container>
      </form>
    </div>
  );
}
