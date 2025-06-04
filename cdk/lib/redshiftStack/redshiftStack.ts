import { Stack, StackProps, RemovalPolicy, aws_s3 as s3, Fn } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as cdk from 'aws-cdk-lib';

export class RedshiftStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    new cdk.CfnOutput(this, '', {
      value: `/`,
      description: '',
      exportName: '',
    });
  }
}
