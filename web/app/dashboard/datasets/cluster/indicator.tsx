import React from 'react';
import { Badge, Indicator } from '@mantine/core';

const IndicatorBadge = ({ color, label, processing = false, textColor }) => {
    return (
      <Indicator
        inline
        size={10}
        offset={7}
        position="middle-start"
        color={color}
        processing={processing}
        withBorder
      >
        <Badge
          variant="outline"
          styles={(theme) => ({
            root: {
              backgroundColor: 'transparent',
              border: 'none',
              paddingLeft: theme.spacing.xl,
              color: textColor, // Use the passed textColor here
            },
          })}
        >
          {label}
        </Badge>
      </Indicator>
    );
  };

export default IndicatorBadge;