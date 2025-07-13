'use client';

import React from 'react';
import {
  Text,
  SimpleGrid,
  Box,
  Button,
  Modal,
  TextInput,
  Group,
  Center,
  rem,
  Title,
} from '@mantine/core';
import Link from 'next/link';
import Image from 'next/image';
import {
  IconBrandNotion,
  IconBrandGoogleDrive,
  IconMail,
  IconBrandAirtable,
  IconJson,
  IconCsv,
  IconPdf,
  IconDatabase,
} from '@tabler/icons-react';
import { useDisclosure } from '@mantine/hooks';
import { modals } from '@mantine/modals';
import { redirect } from 'next/dist/server/api-utils';
import classes from './FeaturesAsymmetrical.module.css';
import AuthenticationTitle from './import';

const Datasource = () => {
  const [opened, { open, close }] = useDisclosure(false);

  return (
    <>
      <Box>
        <Center maw={1000} pt={160}>
          <IconDatabase style={{ width: rem(80), height: rem(80), color: 'tomato' }} />
        </Center>
      </Box>
      <Box>
        <Center maw={1000} mt={1}>
          <Title
            styles={{
              root: {
                fontSize: 80,
              },
            }}
            fw={300}
          >
            Ivory
          </Title>
        </Center>
      </Box>

      <Box maw={600} ml={200}>
        <Text ta="center" fz={18} fw={300} lineClamp={4}>
          Create a dataset from numerous data sources Do low-resource languages have lower quality?
          Low-resource datasets tend to have lower human Low-resource datasets tend to have lower
          human{' '}
        </Text>
      </Box>
      <Center maw={1000} mt={20}>
        <Button onClick={open} color="tomato">
          Add a dataset
        </Button>
      </Center>

      <Modal
        opened={opened}
        onClose={close}
        withCloseButton={false}
        centered
        size="65%"
        radius={20}
        padding={24}
      >
        <AuthenticationTitle onClose={close} />
      </Modal>
    </>
  );
};

export default Datasource;
