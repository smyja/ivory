'use client';

import { Title } from '@mantine/core';
import CardComponent from './card';

export default function Demo() {
  return (
    <>
      <Title order={1} mt={20} mb={20}>
        Settings
      </Title>
      <CardComponent />
    </>
  );
}
